

class FeatureMap:

    def __init__(self) -> None:
        self.state_type_idx_dict = {}

    def get_action_type_idx(self, act_type: str) -> int:
        type_idx_map = {
            "insert-node": 1,
            "update-node": 2,
            "delete-node": 3,
            "move-tree": 4
        }
        return type_idx_map[act_type]
    
    def get_state_type_idx(self, state_type: str) -> int:
        return self.state_type_idx_dict[state_type]

    def set_state_type_dict(self, set_state_type: set) -> None:
        state_type_list = list(set_state_type)
        # sort by dictioanry order
        state_type_list.sort()
        print(f"length of state_type_list: {len(state_type_list)}")
        print(state_type_list)

        for idx, state_type in enumerate(state_type_list):
            self.state_type_idx_dict[state_type] = idx + 5
