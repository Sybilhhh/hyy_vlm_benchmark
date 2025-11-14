import string
from typing import Dict, Any, List, Optional, Tuple
import re
import json
from tqdm import tqdm
from evaluators.evaluators import BaseEvaluator, EvaluationResult
from tabulate import tabulate

from models import OpenAIVLMModel, QwenVLMModel, TarsierVLMModel

_PROMPT_TEMPLATE = string.Template("""
$preamble

$examples

$test_input_output
""".strip())

# We have four steps Entity Extraction, Entity Attribute Extraction, Relation Extraction, and Question Generation
_Entity_Extraction_PREAMBLE = """Task: given input prompts, extract the prompt background, camera and entities from the prompts.
Do not generate same entities multiple times. Do not generate entities that are not present in the prompts. Just output the entities and do not output other things.
output format: Background | "background"
Camera | "camera"
id | entity
""".strip()

_Entity_Attribute_Extraction_PREAMBLE = """Task: given input prompts and entity, extract the attributes of entities from the prompts.
Attributes are intrinsic characteristics of an entity and should not contain external entities that can be divided. Do not generate same attributes multiple times. Do not generate attributes that are not present in the prompts. Do not generate other entities as attributes. If no attribute is present, output "no mention".
output format: entity | attribute | value
"""

_Relation_Extraction_PREAMBLE = """Task: given input prompts and entity, extract the relations between entities from the prompts. Notice that the relations are at least between two entities and if there is only one entity, output "no mention".
Do not generate same relations multiple times. Do not generate relations that are not present in the prompts.
output format: id | entity1 | relation | entity2
"""

_Question_Generation_PREAMBLE = """Task: You are a helpful question generator for video. You are asked to generate questions based on the input video prompts and related entities, attributes and relations. Please ask questions as the format of examples. All the questions may can be answered by yes or no.
output format: question
"""

_Entity_Extraction_EXAMPLE = """Example 1:
input: During harvest, a bear rampages through a cornfield, stalks collapsing in waves.Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
output: 
Background | Harvest 
Camera | no mention
1 | bear
2 | cornfield
3 | stalk
4 | skateboarder
5 | urban skatepark

Example 2:
input: Pink motorcycle weaving through orange traffic cones, camera static.
output: 
Background | city road
Camera | static
1 | motorcycle
2 | traffic cone

Example 3:
input: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
output: 
Background | no mention
Camera | no mention
1 | man
2 | bicycle
3 | hoodie
4 | jeans
"""

_Entity_Extraction_TEST_INPUT_OUTPUT = """Example 4:
input: {prompt}
output: 
"""

_Entity_Attribute_EXAMPLE = """Example 1:
prompt: During harvest, a bear rampages through a cornfield, stalks collapsing in waves. Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
all entities: bear, cornfield, stalk, skateboarder, urban skatepark
entity: bear
attributes: 
bear | number | one

Example 2:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
all entitiess: man, bicycle, hoodie, jeans
entity: man
attiibutes: 
man | number | one
man | age | young
man | hair color | brown
man | hair style | messy
man | scar location | above left eye

Example 3:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
all entities: man, bicycle, hoodie, jeans
entity: hoodie
attributes: 
hoodie | color | blue 

Example 4:
prompt: Under the umbrella, a dancer with metallic skin twirling near a glowing tree
all_entities: umbrella, dancer, tree
entity: umbrella
attributes: 
umbrella | no mention

Example 5:  
prompt: A dancer with metallic skin twirls near a glowing tree.
all_entities: Dancer, tree
entity: Dancer
attributes: 
Dancer | number | one
Dancer | skin | metallic
"""

Entity_Attribute_TEST_INPUT_OUTPUT = """Example 6:
prompt: {prompt}
all_entities: {all_entities}
entity: {entity}
attributes: 
"""

_Relation_Extraction_EXAMPLE = """Example 1:
prompt: During harvest, a bear rampages through a cornfield, stalks collapsing in waves.Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
all_entities: bear, cornfield, stalk, skateboarder, urban skatepark
relations: 1 | bear | rampages | cornfield
2 | stalk | collapsing | cornfield
3 | skateboarder | tearing | urban skatepark 

Example 2:
prompt: Pink motorcycle weaving through orange traffic cones, camera static.
all_entities: motorcycle, traffic cone
relations: 1 | motorcycle | weaving | traffic cone

Example 3:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
all_entities: man, bicycle, hoodie, jeans
relations: 1 | man | riding | bicycle
2 | man | wearing | hoodie
3 | man | wearing | jeans

Example 4:
prompt: a girl is walking forward, /camera push in.
all_entities: girl
relations: no mention
"""

Relation_Extraction_TEST_INPUT_OUTPUT = """Example 5:
prompt: {prompt}
all_entities: {all_entities}
relations: """

Question_Generation_Example = """Example 1:
prompt: During harvest, a bear rampages through a cornfield, stalks collapsing in waves.Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
question_type : entity (entity, attribute, relation)
content: Background | Harvest 
question: Is the video background in the scene of Harvest?

Example 2: 
prompt: Pink motorcycle weaving through orange traffic cones, camera static.
question_type : entity (entity, attribute, relation)
content: 1 | motorcycle
question: Does the video show a motorcycle?

Example 3:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
question_type : attribute (entity, attribute, relation)    
content: man | hair color | brown
question: Is the hair color of the man brown?

Example 4:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
question_type : relation (entity, attribute, relation)
content: 3 | man | riding | bicycle
question: Is the man riding a bicycle?"""


Question_Generation_TEST_INPUT_OUTPUT = """Example 5:
prompt: {prompt}
question_type: {question_type} (entity, attribute, relation)
content: {content}
question: """

_ENTITY_EXTRACTION_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Entity_Extraction_PREAMBLE,
    examples=_Entity_Extraction_EXAMPLE,
    test_input_output=_Entity_Extraction_TEST_INPUT_OUTPUT
)

_ENTITY_ATTRIBUTE_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Entity_Attribute_Extraction_PREAMBLE,
    examples=_Entity_Attribute_EXAMPLE,
    test_input_output=Entity_Attribute_TEST_INPUT_OUTPUT
)

_RELATION_EXTRACTION_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Relation_Extraction_PREAMBLE,
    examples=_Relation_Extraction_EXAMPLE,
    test_input_output=Relation_Extraction_TEST_INPUT_OUTPUT
)

_QUESTION_GENERATION_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Question_Generation_PREAMBLE,
    examples=Question_Generation_Example,
    test_input_output=Question_Generation_TEST_INPUT_OUTPUT
)

class SamplingParams:
    def __init__(self, temperature=0.7, top_p=0.95, max_tokens=150, 
                 repetition_penalty=1.0, stop=None):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.stop = stop or []

async def Build_Scene_Graph(prompt,llm):
    #sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=150, repetition_penalty=1.2,stop=["Example","Input","Task","Entity"])
    # Task 1: Entity Extraction
    task1_prompt = _ENTITY_EXTRACTION_PROMPT.format(prompt=prompt)
    task1_outputs = await llm._predict_one_sample(task1_prompt)
    # Task 2: Entity Attribute Extraction
    all_entity_list = [item for item in task1_outputs.split("\n") if "|" in item][2:]
    entity_list = [item for item in task1_outputs.split("\n") if "|" in item]
    all_entities = ", ".join([item.split("|")[1].strip() for item in all_entity_list])
    task2_prompts = []
    for entity in entity_list:
        entity = entity.split("|")[1].strip()
        if entity == "no mention":
            continue
        task2_prompts.append(_ENTITY_ATTRIBUTE_PROMPT.format(prompt=prompt, all_entities=all_entities,entity=entity))
    
    task2_outputs = []
    for task2_prompt in task2_prompts:
        task2_outputs.append(await llm._predict_one_sample(task2_prompt))
    # task2_outputs = [output.outputs[0].text for output in task2_outputs]
    # Task 3: Relation Extraction
    task3_prompt = _RELATION_EXTRACTION_PROMPT.format(prompt=prompt, all_entities=all_entities)
    task3_outputs = await llm._predict_one_sample(task3_prompt)
    # task3_outputs = [task3_outputs]
    entity = [entity for entity in task1_outputs.split("\n") if "|" in entity]
    attributes = []
    for output in task2_outputs:
        attributes += [attribute for attribute in output.split("\n") if "|" in attribute]


    relations = [relation for relation in task3_outputs.split("\n") if "|" in relation]
    return entity, attributes, relations

async def Question_Generation(entity, attributes, relations, prompt, llm):
    # sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=150, repetition_penalty=1.2,stop=["Example","Input","Task","Entity"])
    # Question Generation entity
    entity_outputs = []
    for item in entity:
        if "no mention" in item.lower():
            continue
        entity_prompt = _QUESTION_GENERATION_PROMPT.format(prompt=prompt, question_type="entity", content=item)
        entity_outputs.append(await llm._predict_one_sample(entity_prompt))
    attribute_outputs = []
    for item in attributes:
        if "no mention" in item.lower():
            continue
        attribute_prompt = _QUESTION_GENERATION_PROMPT.format(prompt=prompt, question_type="attribute", content=item)
        attribute_outputs.append(await llm._predict_one_sample(attribute_prompt))
    relation_outputs = []
    for item in relations:
        if "no mention" in item.lower():
            continue
        relation_prompt = _QUESTION_GENERATION_PROMPT.format(prompt=prompt, question_type="relation", content=item)
        relation_outputs.append(await llm._predict_one_sample(relation_prompt))
    # entity_outputs = await llm._predict_one_sample(entity_prompts)
    # attribute_outputs = await llm._predict_one_sample(attribute_prompts)
    # relation_outputs = await llm._predict_one_sample(relation_prompts)

    questions = entity_outputs + attribute_outputs + relation_outputs
    return questions


class ETVAEvaluator(BaseEvaluator):
    """Evaluator for Scene Graph based Question Answering on videos."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize LLM and VLM models
        self.llm = OpenAIVLMModel(config.get("llm", {}))
        self.vlm = QwenVLMModel(config.get('vlm-qwen', {}))
        # self.vlm = TarsierVLMModel(config.get('vlm-tarsier', {}))
        
        # Question categories
        self.question_types = ["entity", "attribute", "relation", "custom"]
        
        # Base prompt template for VLM
        self.base_prompt_template = config.get(
            'base_prompt_template', 
            "Watch the video thoroughly, and answer the question based on video content with yes or no. {}"
        )
        
        # Additional custom questions (if any)
        self.custom_questions = config.get('custom_questions', [])
        
        # Evaluation settings
        self.include_scene_graph = config.get('include_scene_graph', True)
        self.verbose = config.get('verbose', False)

    async def _ensure_lms(self):
        if not await self.llm.is_initialized():
            await self.llm.load_model()
        await self.vlm.load_model()
    
    async def evaluate(self, predictions: List[Dict[str, Any]], dataset) -> EvaluationResult:
        """Main evaluation entry point."""
        await self._ensure_lms()
        
        # Process each sample
        all_results = []
        
        for sample in tqdm(predictions, desc="Evaluating samples"):
            sample_result = await self._evaluate_sample(sample)
            all_results.append(sample_result)
        
        # Aggregate results by question type
        aggregated_results = self._aggregate_results(all_results)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(aggregated_results)
        
        # Generate summary table
        summary_table = self._generate_summary_table(aggregated_results)
        
        
        return EvaluationResult(
            score=overall_metrics['overall_accuracy'],
            details={
                'aggregated_results': aggregated_results,
                'overall_metrics': overall_metrics,
                'summary_table': summary_table,
                'total_samples': len(predictions),
                'total_questions': overall_metrics['total_questions']
            },
            method='scene_graph_qa',
            ground_truth_required=False
        )
    
    async def _evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample (prompt + video)."""
        
        reference = sample.get('reference', '')
        video_path = sample.get('video_path', '')
        sample_id = sample.get('id', 'unknown')
        
        result = {
            'sample_id': sample_id,
            'reference': reference,
            'video_path': video_path,
            'questions': [],
            'scene_graph': {}
        }
        
        try:
            # Step 1: Build scene graph from prompt
            if self.include_scene_graph:
                entity, attributes, relations = await Build_Scene_Graph(reference, self.llm)
                result['scene_graph'] = {
                    'entities': entity,
                    'attributes': attributes,
                    'relations': relations
                }
                
                # Step 2: Generate questions based on scene graph
                questions = await Question_Generation(
                    entity, attributes, relations, reference, self.llm
                )
            else:
                questions = []
            
            # Add custom questions if provided
            if self.custom_questions:
                questions.extend(self.custom_questions)
            
            # For sample-specific custom questions
            if 'custom_questions' in sample:
                questions.extend(sample['custom_questions'])

            # Delete duplicate qustions
            questions = list(set(questions))
            
            # Step 3: Answer each question using VLM
            for question in questions:
                question_result = await self._answer_question(
                    question, video_path
                )
                result['questions'].append(question_result)
            
            result['status'] = 'success'
            
        except Exception as e:
            self.logger.error(f"Error evaluating sample {sample_id}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    async def _answer_question(self, question: str, video_path: str) -> Dict[str, Any]:
        """Answer a single question about a video."""
        
        # Format question with base prompt
        formatted_question = self.base_prompt_template.format(question)
        
        try:
            # Get answer from VLM
            answer = await self.vlm._predict_one_sample(formatted_question, video_path=video_path)
            
            # Parse yes/no answer
            parsed_answer = self._parse_yes_no_answer(answer)
            
            return {
                'question': question,
                'raw_answer': answer,
                'parsed_answer': parsed_answer,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'question': question,
                'raw_answer': None,
                'parsed_answer': None,
                'status': 'error',
                'error': str(e)
            }
    
    def _parse_yes_no_answer(self, answer: str) -> Optional[bool]:
        """Parse VLM answer to extract yes/no response."""
        
        if not answer:
            return None
        
        answer_lower = answer.lower().strip()
        
        # Look for yes/no patterns
        yes_patterns = [r'\byes\b', r'\btrue\b', r'\bcorrect\b', r'\baffirmative\b']
        no_patterns = [r'\bno\b', r'\bfalse\b', r'\bincorrect\b', r'\bnegative\b']
        
        for pattern in yes_patterns:
            if re.search(pattern, answer_lower):
                return True
        
        for pattern in no_patterns:
            if re.search(pattern, answer_lower):
                return False
        
        # If no clear yes/no found
        return None
    
    def _aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all questions."""
        
        aggregated = {
            'total': 0,
            'answered': 0,
            'yes': 0,
            'no': 0,
            'unclear': 0,
            'per_sample': []
        }
        
        for result in all_results:
            if result['status'] != 'success':
                continue
            
            sample_stats = {
                'sample_id': result['sample_id'],
                'total': 0,
                'answered': 0,
                'yes': 0,
                'no': 0,
                'unclear': 0
            }
            
            for q_result in result['questions']:
                aggregated['total'] += 1
                sample_stats['total'] += 1
                
                if q_result['status'] == 'success':
                    aggregated['answered'] += 1
                    sample_stats['answered'] += 1
                    
                    if q_result['parsed_answer'] is True:
                        aggregated['yes'] += 1
                        sample_stats['yes'] += 1
                    elif q_result['parsed_answer'] is False:
                        aggregated['no'] += 1
                        sample_stats['no'] += 1
                    else:
                        aggregated['unclear'] += 1
                        sample_stats['unclear'] += 1
            
            aggregated['per_sample'].append(sample_stats)
        
        # Calculate overall percentages
        if aggregated['total'] > 0:
            aggregated['answer_rate'] = aggregated['answered'] / aggregated['total']
            aggregated['yes_rate'] = aggregated['yes'] / aggregated['total']
            aggregated['no_rate'] = aggregated['no'] / aggregated['total']
            aggregated['unclear_rate'] = aggregated['unclear'] / aggregated['total']
            aggregated['clarity_rate'] = (aggregated['yes'] + aggregated['no']) / aggregated['total']
        else:
            aggregated['answer_rate'] = 0
            aggregated['yes_rate'] = 0
            aggregated['no_rate'] = 0
            aggregated['unclear_rate'] = 0
            aggregated['clarity_rate'] = 0
        
        return aggregated
    
    def _calculate_overall_metrics(self, aggregated_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall metrics across all questions."""
        
        return {
            'total_questions': aggregated_results['total'],
            'total_answered': aggregated_results['answered'],
            'overall_answer_rate': round(aggregated_results['answer_rate'], 4),
            'overall_clarity_rate': round(aggregated_results['clarity_rate'], 4),
            'overall_accuracy': round(aggregated_results['clarity_rate'], 4),  # Use clarity as proxy for accuracy
            'yes_ratio': round(aggregated_results['yes_rate'], 4),
            'no_ratio': round(aggregated_results['no_rate'], 4),
            'unclear_ratio': round(aggregated_results['unclear_rate'], 4)
        }
    
    def _generate_summary_table(self, aggregated_results: Dict[str, Any]) -> str:
        """Generate summary table for results."""
        
        table_data = [['Metric', 'Value']]
        
        table_data.append(['Total Questions', str(aggregated_results['total'])])
        table_data.append(['Questions Answered', str(aggregated_results['answered'])])
        table_data.append(['Yes Responses', str(aggregated_results['yes'])])
        table_data.append(['No Responses', str(aggregated_results['no'])])
        table_data.append(['Unclear Responses', str(aggregated_results['unclear'])])
        table_data.append(['Answer Rate', f"{aggregated_results['answer_rate']:.2%}"])
        table_data.append(['Clarity Rate', f"{aggregated_results['clarity_rate']:.2%}"])
        table_data.append(['Yes Ratio', f"{aggregated_results['yes_rate']:.2%}"])
        table_data.append(['No Ratio', f"{aggregated_results['no_rate']:.2%}"])
        table_data.append(['Unclear Ratio', f"{aggregated_results['unclear_rate']:.2%}"])
        
        # Add per-sample breakdown if available
        if aggregated_results.get('per_sample'):
            table_data.append(['', ''])  # Empty row for separation
            table_data.append(['Per Sample Breakdown', ''])
            for sample_stats in aggregated_results['per_sample']:
                table_data.append([
                    f"  Sample {sample_stats['sample_id']}", 
                    f"{sample_stats['answered']}/{sample_stats['total']} answered"
                ])
        
        return tabulate(table_data, headers='firstrow', tablefmt='grid')