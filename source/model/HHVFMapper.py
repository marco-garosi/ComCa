from .VFMapper import VFMapper


class HHVFMapper(VFMapper):
    """Hard match, hard assignment mapper

    Hard match: match prediction to ground truth based on a predefined, rule-based criteria
    Hard assignment: assign to the ground truths that match based on the chosen criteria
        If more than a match is found, they have the same importance
        A match is a plain match
    """

    def __init__(self, ground_truth: list[str], mode: str = 'exact') -> None:
        super().__init__(ground_truth)

        assert mode in ['exact', 'case_insensitive', 'contained', 'contained_case_insensitive']
        self.mode = mode

        if self.mode in ['case_insensitive', 'contained_case_insensitive']:
            self.ground_truth = {
                k.lower(): v
                for k, v in self.ground_truth.items()
            }

    def __call__(self, prediction: list[str]) -> list[list[int]]:
        if self.mode == 'exact':
            return self.match_exact(prediction)
        
        if self.mode == 'case_insensitive':
            return self.match_case_insensitive(prediction)
        
        if self.mode == 'contained':
            return self.match_contained(prediction)
        
        if self.mode == 'contained_case_insensitive':
            return self.match_contained_case_insensitive(prediction)
    
    def match_exact(self, prediction: list[str]) -> list[list[int]]:
        wrap = lambda x: [x] if x is not None else []
        return [
            wrap(self.ground_truth.get(pred))
            for pred in prediction
        ]
    
    def match_case_insensitive(self, prediction: list[str]) -> list[list[int]]:
        prediction = [x.lower() for x in prediction]
        return self.match_exact(prediction)
    
    def match_contained(self, prediction: list[str]) -> list[list[int]]:
        return [
            [idx for idx, gt in enumerate(self.ground_truth.keys()) if gt.__contains__(p)]
            for p in prediction
        ]
    
    def match_contained_case_insensitive(self, prediction: list[str]) -> list[list[int]]:
        prediction = [x.lower() for x in prediction]
        return self.match_contained(prediction)



# Tests
if __name__ == '__main__':
    def get_test_1():
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

        mapping = [[0], [1], [2], [3], [4]]

        return ground_truth, predictions, mapping, 'exact'
    
    def get_test_2():
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

        mapping = [[0], [2], [3], [4]]

        return ground_truth, predictions, mapping, 'exact'
    
    def get_test_3():
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

        mapping = [[0], [], [2], [3], [4]]

        return ground_truth, predictions, mapping, 'exact'
    
    def get_test_4():
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

        mapping = [[0], [], [1], [2], [3]]

        return ground_truth, predictions, mapping, 'exact'
    
    def get_test_5():
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

        mapping = [[0], [], [1], [2], [3]]

        return ground_truth, predictions, mapping, 'case_insensitive'

    def get_test_6():
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

        mapping = [[0], [], [1], [2], [3]]

        return ground_truth, predictions, mapping, 'contained'

    def get_test_7():
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

        mapping = [[0], [], [1], [2], [3]]

        return ground_truth, predictions, mapping, 'contained_case_insensitive'

    def get_test(idx):
        if idx == 1:
            return get_test_1()
        if idx == 2:
            return get_test_2()
        if idx == 3:
            return get_test_3()
        if idx == 4:
            return get_test_4()
        if idx == 5:
            return get_test_5()
        if idx == 6:
            return get_test_6()
        if idx == 7:
            return get_test_7()
        
        return None, None, None, None

    for idx in range(7):
        idx += 1
        ground_truth, predictions, gt_mapping, mode = get_test(idx)
        mapper = HHVFMapper(ground_truth, mode=mode)
        mapping = mapper(predictions)
        
        if mapping == gt_mapping:
            print(f'√ Test {idx} passed successfully')
        else:
            print(mapping)
            print(gt_mapping)
            print(f'X Test {idx} failed')
