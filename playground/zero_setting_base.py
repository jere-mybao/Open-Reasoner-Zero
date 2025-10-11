from typing import List

from jinja2 import Template

from orz.ppo import PromptDataset


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):
        prompt_template_jinja = """\
{{bos_token}}You are an expert reasoning evaluator. Your task is to meticulously examine another model's reasoning trace, step-by-step logic, and inferential patterns to determine whether its final answer is more likely to be correct or incorrect.

You will be provided with:
1. An original question
2. Another model's reasoning process and answer with confidence in the format {answer, confidence}

Focus your evaluation on:
1. **Logical Structure**: Are the reasoning steps logically connected and valid?
2. **Evidence Quality**: How strong and reliable are the sources and claims presented?
3. **Inferential Gaps**: Are there missing steps or unjustified leaps in logic?
4. **Consistency**: Does the reasoning remain internally consistent throughout?
5. **Confidence Calibration**: Is the stated confidence level appropriate given the evidence?

Your evaluation process should be:
1. Trace through each reasoning step systematically
2. Identify specific strengths and weaknesses in the logical flow
3. Assess the quality of evidence and sources cited
4. Evaluate whether conclusions follow from premises
5. Consider alternative explanations or counterarguments

Your analysis should be enclosed within <think> </think> tags, and your final judgment should be in <answer> </answer> tags using only:
- 1 if the first model's answer is more likely to be correct
- 0 if the first model's answer is more likely to be incorrect

User: {{prompt}}
Assistant: <think>\
"""
        prompt_instruction_template_jinja = """\
Analyze the following model's step-by-step reasoning process. Focus on the logical flow, evidence quality, and inferential validity rather than independently solving the problem.

{{prompt}}
"""

        assert len(dialogue) == 2, "dialogue must contain 2 items"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        extra = {"answer": dialogue[1]["ground_truth"]["value"]}

        return prompt, extra


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        prompt_template_jinja = """\
{{bos_token}}You are an expert reasoning evaluator. Your task is to meticulously examine another model's reasoning trace, step-by-step logic, and inferential patterns to determine whether its final answer is more likely to be correct or incorrect.

You will be provided with:
1. An original question
2. Another model's reasoning process and answer with confidence in the format {answer, confidence}

Focus your evaluation on:
1. **Logical Structure**: Are the reasoning steps logically connected and valid?
2. **Evidence Quality**: How strong and reliable are the sources and claims presented?
3. **Inferential Gaps**: Are there missing steps or unjustified leaps in logic?
4. **Consistency**: Does the reasoning remain internally consistent throughout?
5. **Confidence Calibration**: Is the stated confidence level appropriate given the evidence?

Your evaluation process should be:
1. Trace through each reasoning step systematically
2. Identify specific strengths and weaknesses in the logical flow
3. Assess the quality of evidence and sources cited
4. Evaluate whether conclusions follow from premises
5. Consider alternative explanations or counterarguments

Your analysis should be enclosed within <think> </think> tags, and your final judgment should be in <answer> </answer> tags using only:
- 1 if the first model's answer is more likely to be correct
- 0 if the first model's answer is more likely to be incorrect

User: {{prompt}}
Assistant: <think>\
"""
        prompt_instruction_template_jinja = """\
Analyze the following model's step-by-step reasoning process. Focus on the logical flow, evidence quality, and inferential validity rather than independently solving the problem.

{{prompt}}
"""
        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue["prompt"][0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

        return prompt, extra
