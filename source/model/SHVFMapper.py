from typing import Callable, Optional

from .VFMapper import VFMapper
import torch
import torch.nn.functional as F
import numpy as np

from ..cache_generation import parse_attribute_from_string


class SHVFMapper(VFMapper):
    """Soft match, hard assignment mapper

    Soft match: match prediction to ground truth based on an embedding model
    Hard assignment: assign to the best-matching ground truth
    """

    _ground_truth_embeddings: Optional[torch.Tensor] = None

    def __init__(self,
                 ground_truth: list[str],
                 embed: Callable[[list[str]], torch.Tensor],
                 prompt_mode: Optional[str] = None,
                 templates: Optional[dict[str, dict[str, list[str]]]] = None,
                 template: Optional[str] = None,
                 noun: Optional[str] = None,
                 idx2is_has: Optional[dict[int, str]] = None,
                 default_is_has: Optional[str] = 'is',
                 threshold: Optional[float] = None,
                 epsilon: Optional[float] = 1e-5,
                ) -> None:
        super().__init__(ground_truth)

        assert prompt_mode in [None, 'ovad_style']
        assert prompt_mode is None or templates is not None
        assert prompt_mode is None or template is not None
        assert prompt_mode is None or noun is not None
        assert prompt_mode is None or idx2is_has is not None

        self.embed = embed
        self.prompt_mode = prompt_mode
        self.templates = templates
        self.template = template
        self.noun = noun
        self.idx2is_has = idx2is_has
        self.default_is_has = default_is_has
        self.threshold = threshold
        self.epsilon = epsilon

        if self.threshold is not None and self.epsilon is not None:
            self.threshold -= self.epsilon

        self._ground_truth_embeddings = self.template_and_embed(list(self.ground_truth.keys()), self.idx2is_has).unsqueeze(1)

    def get_prompts(self, text: list[str], idx2is_has: Optional[dict[int, str]] = None):
        if self.prompt_mode is None:
            return text
        
        if self.prompt_mode == 'ovad_style':
            prompts = []

            for idx, x in enumerate(text):
                try:
                    group, synonyms = parse_attribute_from_string(x)
                except:
                    # Cannot split into group and synonyms, so return plaint text
                    return text

                if idx2is_has is not None:
                    is_has = idx2is_has[int(idx)]
                else:
                    is_has = self.default_is_has
                
                prompt = np.random.choice(self.templates[is_has][self.template])
                prompts.append(prompt.format(attr=', '.join(synonyms), noun=self.noun, dobj=group))

            return prompts

    def template_and_embed(self, text: list[str], idx2is_has: Optional[dict[int, str]] = None) -> torch.Tensor:
        prompts = self.get_prompts(text, idx2is_has=idx2is_has)
        
        with torch.no_grad():
            return self.embed(prompts)

    def __call__(self, text: list[str], idx2is_has: Optional[dict[int, str]] = None) -> list[list[int]]:
        embeddings = self.template_and_embed(text, idx2is_has)
        
        # Compute the cosine similarity with the ground truth
        # (P, C) @ (GT, 1, C) --> (P, GT, C) --> (P, GT)
        similarity = F.cosine_similarity(embeddings, self._ground_truth_embeddings, dim=-1).T

        # Map to ground truth
        if self.threshold is not None:
            # (P, GT) --(thresholding)--> (1, P)
            mapping = [[] for _ in range(len(text))]

            mask = similarity >= self.threshold
            for idx in torch.any(mask, dim=-1).nonzero().view(-1):
                mapping[idx] = mask[idx].nonzero().view(-1).cpu().tolist()

            # Functionally, the code above is the same as the following.
            # However, the following piece of code is slower and exploits less
            # the GPU parallel processing capabilities
            # mapping = []
            # for idx, pred in enumerate(similarity):
            #     if torch.all(pred < self.threshold):
            #         mapping.append([])
            #         continue
            #     mapping.append((pred >= self.threshold).nonzero().view(-1).cpu().tolist())

        else:
            # (P, GT) --(argmax)--> (P) --(view)--> (1, P)
            mapping = similarity.argmax(dim=-1).view(-1, 1).cpu().tolist()

        return mapping


# Tests
if __name__ == '__main__':
    from .SentenceBert import SentenceBERT
    from ..utils import load_templates_cache_generation
    from ..utils import load_annotations
    import config

    embed = SentenceBERT()
    
    template_file = 'ovad.json'
    templates = load_templates_cache_generation(template_file, base_dir=config.BASE_PATH)
    noun = 'object'

    annotations = load_annotations(config.BASE_PATH)


    def get_test_1(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: blue',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'color: red',
            'color: blue',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        mapping = {
            'options': [
                [[0], [1], [2], [3], [4]],
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }
    
    def get_test_2(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: blue',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'color: red',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        mapping = {
            'options': [
                [[0], [2], [3], [4]],
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }
    
    def get_test_3(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: blue',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'color: red',
            'color: green',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        mapping = {
            'options': [
                [[0], [], [2], [3], [4]],  # this method cannot do this, unless we do thresholding
                [[0], [0], [2], [3], [4]], # green -> red: acceptable
                [[0], [1], [2], [3], [4]], # green -> blue: acceptable
                [[0], [2], [2], [3], [4]], # green -> black: acceptable
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }
    
    def get_test_4(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'color: red',
            'color: green',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        mapping = {
            'options': [
                [[0], [], [1], [2], [3]],  # this method cannot do this, unless we do thresholding
                [[0], [0], [1], [2], [3]], # green -> red: acceptable
                [[0], [1], [1], [2], [3]], # green -> black: acceptable
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }
    
    def get_test_5(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'Color: red',
            'color: green',
            'COLOR: black',
            'material: copper/metal',
            'length: short',
        ]

        mapping = {
            'options': [
                [[0], [], [1], [2], [3]],  # this method cannot do this, unless we do thresholding
                [[0], [0], [1], [2], [3]], # green -> red: acceptable
                [[0], [1], [1], [2], [3]], # green -> black: acceptable
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }

    def get_test_6(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'color: red',
            'green',
            'color: black',
            'metal',
            'short',
        ]

        mapping = {
            'options': [
                [[0], [], [1], [2], [3]],  # this method cannot do this, unless we do thresholding
                [[0], [0], [1], [2], [3]], # green -> red: acceptable
                [[0], [1], [1], [2], [3]], # green -> black: acceptable
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }

    def get_test_7(prompt_mode: Optional[str] = None, default_is_has: str = 'is'):
        ground_truth = [
            'color: red',
            'color: black',
            'material: copper/metal',
            'length: short',
        ]

        predictions = [
            'Color: red',
            'Green',
            'COLOR: black',
            'metal',
            'SHORT',
        ]

        mapping = {
            'options': [
                [[0], [], [1], [2], [3]],  # this method cannot do this, unless we do thresholding
                [[0], [0], [1], [2], [3]], # green -> red: acceptable
                [[0], [1], [1], [2], [3]], # green -> black: acceptable
            ]
        }

        return ground_truth, predictions, mapping, {
            'prompt_mode': prompt_mode,
            'default_is_has': default_is_has,
        }

    def get_test(idx, prompt_mode, default_is_has):
        if idx == 1:
            return get_test_1(prompt_mode=prompt_mode, default_is_has=default_is_has)
        if idx == 2:
            return get_test_2(prompt_mode=prompt_mode, default_is_has=default_is_has)
        if idx == 3:
            return get_test_3(prompt_mode=prompt_mode, default_is_has=default_is_has)
        if idx == 4:
            return get_test_4(prompt_mode=prompt_mode, default_is_has=default_is_has)
        if idx == 5:
            return get_test_5(prompt_mode=prompt_mode, default_is_has=default_is_has)
        if idx == 6:
            return get_test_6(prompt_mode=prompt_mode, default_is_has=default_is_has)
        if idx == 7:
            return get_test_7(prompt_mode=prompt_mode, default_is_has=default_is_has)
        
        return None, None, None, None

    for prompt_mode in [None, 'ovad_style']:
        print('=' * 89)
        print(f'Prompt mode: {prompt_mode}')

        for default_is_has in ['is', 'has']:
            print('\n' + '=' * (89 // 2))
            print(f'Default is/has: {default_is_has}')

            for template in ['none', 'a', 'the', 'photo']:
                print('\n' + '=' * (89 // 2))
                print(f'Template: {template}')

                for idx in range(7):
                    idx += 1
                    ground_truth, predictions, gt_mapping, args = get_test(idx, prompt_mode=prompt_mode, default_is_has=default_is_has)
                    mapper = SHVFMapper(
                        ground_truth,
                        embed=embed,
                        prompt_mode=args['prompt_mode'],
                        templates=templates,
                        template=template,
                        noun=noun,
                        idx2is_has=annotations['idx2is_has'],
                        default_is_has=args['default_is_has'],
                    )
                    mapping = mapper(predictions)

                    gt_mapping = gt_mapping['options']
                    
                    # Using the `in` operator as `gt_mapping` is a list of candidate (acceptable)
                    # solutions, therefore the `mapping` is fine/correct if it is contained
                    # in `gt_mapping`
                    if mapping in gt_mapping:
                        print(f'√ Test {idx} passed successfully')
                    else:
                        print(f'X Test {idx} failed')
                        print(f'\t== Debug information ==')
                        print(f'\t• Prediction:   {mapping}')
                        print(f'\t• Ground truth: {gt_mapping}')

        print()
