from ..core.loader import load_model, hash_model_file
from ..core.loader.file import load_keys_dict
from ..core.vram import AutoWrappedModule
from ..configs import MODEL_CONFIGS, VRAM_MANAGEMENT_MODULE_MAPS
import importlib, json, torch
import re


class ModelPool:
    def __init__(self):
        self.model = []
        self.model_name = []
        self.model_path = []
        
    def import_model_class(self, model_class):
        split = model_class.rfind(".")
        model_resource, model_class = model_class[:split], model_class[split+1:]
        model_class = importlib.import_module(model_resource).__getattribute__(model_class)
        return model_class
    
    def need_to_enable_vram_management(self, vram_config):
        return vram_config["offload_dtype"] is not None and vram_config["offload_device"] is not None
    
    def fetch_module_map(self, model_class, vram_config):
        if self.need_to_enable_vram_management(vram_config):
            if model_class in VRAM_MANAGEMENT_MODULE_MAPS:
                module_map = {self.import_model_class(source): self.import_model_class(target) for source, target in VRAM_MANAGEMENT_MODULE_MAPS[model_class].items()}
            else:
                module_map = {self.import_model_class(model_class): AutoWrappedModule}
        else:
            module_map = None
        return module_map
    
    def load_model_file(self, config, path, vram_config, vram_limit=None):
        model_class = self.import_model_class(config["model_class"])
        model_config = config.get("extra_kwargs", {})
        if "state_dict_converter" in config:
            state_dict_converter = self.import_model_class(config["state_dict_converter"])
        else:
            state_dict_converter = None
        module_map = self.fetch_module_map(config["model_class"], vram_config)
        model = load_model(
            model_class, path, model_config,
            vram_config["computation_dtype"], vram_config["computation_device"],
            state_dict_converter,
            use_disk_map=True,
            vram_config=vram_config, module_map=module_map, vram_limit=vram_limit,
        )
        return model
    
    def default_vram_config(self):
        vram_config = {
            "offload_dtype": None,
            "offload_device": None,
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cpu",
        }
        return vram_config
    
    def auto_load_model(self, path, vram_config=None, vram_limit=None, clear_parameters=False):
        print(f"Loading models from: {json.dumps(path, indent=4)}")
        if vram_config is None:
            vram_config = self.default_vram_config()
        model_hash = hash_model_file(path)
        loaded = False
        for config in MODEL_CONFIGS:
            if config["model_hash"] == model_hash:
                model = self.load_model_file(config, path, vram_config, vram_limit=vram_limit)
                if clear_parameters: self.clear_parameters(model)
                self.model.append(model)
                model_name = config["model_name"]
                self.model_name.append(model_name)
                self.model_path.append(path)
                model_info = {"model_name": model_name, "model_class": config["model_class"], "extra_kwargs": config.get("extra_kwargs")}
                print(f"Loaded model: {json.dumps(model_info, indent=4)}")
                loaded = True
        if not loaded:
            # Fallback: some fine-tuned checkpoints (especially Diffusers / DDP exports) keep the
            # *same architecture* but change key prefixes or naming, which breaks our hash-based
            # detection. We try to infer the most likely config from tensor key patterns + shapes.
            inferred_configs = self._infer_model_configs_by_keys(path)
            for config in inferred_configs:
                model = self.load_model_file(config, path, vram_config, vram_limit=vram_limit)
                if clear_parameters: self.clear_parameters(model)
                self.model.append(model)
                model_name = config["model_name"]
                self.model_name.append(model_name)
                self.model_path.append(path)
                model_info = {"model_name": model_name, "model_class": config["model_class"], "extra_kwargs": config.get("extra_kwargs")}
                print(f"[Fallback] Loaded model: {json.dumps(model_info, indent=4)}")
                loaded = True
            if not loaded:
                raise ValueError(f"Cannot detect the model type. File: {path}. Model hash: {model_hash}")

    def _infer_model_configs_by_keys(self, path):
        """
        Infer model config(s) when the key+shape hash isn't recognized.
        Currently focuses on WanVideo DiT sharded checkpoints (diffusion_pytorch_model*.safetensors).
        Returns a list of MODEL_CONFIG-like dicts.
        """
        try:
            keys_dict = load_keys_dict(path)
        except Exception as e:
            print(f"[Fallback] Failed to inspect model keys for {path}: {e}")
            return []

        if not keys_dict:
            return []

        keys = list(keys_dict.keys())
        keyset = set(keys)

        def _has_any(substrings):
            return any(any(s in k for s in substrings) for k in keys)

        # ---- WanVideo DiT inference ----
        # Detect patch embedding tensor to infer (dim, in_dim).
        #
        # Canonical key in this repo ends with `patch_embedding.weight`, but Diffusers / custom exports
        # may rename it (e.g. `patch_embed.proj.weight`). To keep this robust, fall back to detecting
        # a Conv3D weight with kernel/stride matching Wan patch embedding: (1, 2, 2).
        patch_key = None
        for k in keys:
            if k.endswith("patch_embedding.weight"):
                patch_key = k
                break
        if patch_key is None:
            conv3d_candidates = []
            for k, shape in keys_dict.items():
                if not (isinstance(shape, (list, tuple)) and len(shape) == 5):
                    continue
                # Wan patch embedding uses kernel_size=(1,2,2) so weight has trailing [1,2,2].
                if shape[2] == 1 and shape[3] == 2 and shape[4] == 2:
                    conv3d_candidates.append((k, shape))
            if conv3d_candidates:
                # Prefer keys that "look like" patch embedding; otherwise pick the largest out_dim.
                conv3d_candidates.sort(
                    key=lambda kv: (
                        ("patch" not in kv[0].lower()),
                        -int(kv[1][0]),
                    )
                )
                patch_key, _ = conv3d_candidates[0]
        if patch_key is None:
            return []
        patch_shape = keys_dict.get(patch_key)
        if not (isinstance(patch_shape, (list, tuple)) and len(patch_shape) >= 2):
            return []
        dim = patch_shape[0]
        in_dim = patch_shape[1]

        # Infer number of transformer blocks
        block_idxs = []
        for k in keys:
            m = re.search(r"(?:^|\.)(?:blocks)\.(\d+)\.", k)
            if m:
                block_idxs.append(int(m.group(1)))
        num_layers = (max(block_idxs) + 1) if block_idxs else None

        # Heuristic: Diffusers exports often contain attn1/attn2 naming + condition_embedder
        is_diffusers_like = _has_any(["attn1.", "attn2.", "condition_embedder."])

        # Heuristic: DDP / Lightning checkpoints often prefix keys with `model.` or `module.`
        has_common_prefix = any(k.startswith(("model.", "module.")) for k in keys)

        # Search best matching Wan DiT config by (dim, in_dim, num_layers)
        candidates = []
        for cfg in MODEL_CONFIGS:
            if cfg.get("model_name") != "wan_video_dit":
                continue
            if cfg.get("model_class") != "diffsynth.models.wan_video_dit.WanModel":
                continue
            extra = cfg.get("extra_kwargs") or {}
            if extra.get("dim") != dim:
                continue
            if extra.get("in_dim") != in_dim:
                continue
            if num_layers is not None and extra.get("num_layers") != num_layers:
                continue
            candidates.append(cfg)

        # If we couldn't find an exact match, relax to (dim, num_layers) only (common when in_dim differs by adapters)
        if not candidates and num_layers is not None:
            for cfg in MODEL_CONFIGS:
                if cfg.get("model_name") != "wan_video_dit":
                    continue
                if cfg.get("model_class") != "diffsynth.models.wan_video_dit.WanModel":
                    continue
                extra = cfg.get("extra_kwargs") or {}
                if extra.get("dim") != dim:
                    continue
                if extra.get("num_layers") != num_layers:
                    continue
                candidates.append(cfg)

        if not candidates:
            return []

        # Prefer a converter that matches the checkpoint naming style
        def _converter_name(c):
            return str(c.get("state_dict_converter") or "")

        if is_diffusers_like:
            preferred = [c for c in candidates if _converter_name(c).endswith("WanVideoDiTFromDiffusers")]
            candidates = preferred or candidates

        # Copy and patch converter when needed
        out = []
        for cfg in candidates[:1]:
            cfg2 = dict(cfg)
            # If we matched a candidate only by (dim, num_layers), its `in_dim` may differ from the
            # checkpoint (common for TV2V "channel_real" style conditioning which doubles channels).
            # Patch `extra_kwargs.in_dim` to the inferred value so model instantiation matches the checkpoint.
            extra2 = dict(cfg.get("extra_kwargs") or {})
            extra2["in_dim"] = in_dim
            cfg2["extra_kwargs"] = extra2
            if is_diffusers_like:
                cfg2.setdefault("state_dict_converter", "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTFromDiffusers")
            elif has_common_prefix:
                # Strip `model.` / `module.` prefixes if present.
                cfg2.setdefault("state_dict_converter", "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter")
            out.append(cfg2)

        print(
            "[Fallback] Inferred Wan DiT config from keys: "
            + json.dumps(
                {
                    "dim": dim,
                    "in_dim": in_dim,
                    "num_layers": num_layers,
                    "diffusers_like": is_diffusers_like,
                    "candidates": [c.get("extra_kwargs", {}) for c in out],
                },
                indent=4,
            )
        )
        return out
    
    def fetch_model(self, model_name, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available. This is not an error.")
            model = None
        elif len(fetched_models) == 1:
            print(f"Using {model_name} from {json.dumps(fetched_model_paths[0], indent=4)}.")
            model = fetched_models[0]
        else:
            if index is None:
                model = fetched_models[0]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths[0], indent=4)}.")
            elif isinstance(index, int):
                model = fetched_models[:index]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths[:index], indent=4)}.")
            else:
                model = fetched_models
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {model_name} from {json.dumps(fetched_model_paths, indent=4)}.")
        return model

    def clear_parameters(self, model: torch.nn.Module):
        for name, module in model.named_children():
            self.clear_parameters(module)
        for name, param in model.named_parameters(recurse=False):
            setattr(model, name, None)
