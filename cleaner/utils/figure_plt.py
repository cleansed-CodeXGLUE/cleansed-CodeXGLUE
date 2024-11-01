

import matplotlib.pyplot as plt
from venn import venn


def venn_plt(**kwargs):
    """
    Function to plot venn diagram
    """
    # list -> set
    sets = {k: set(v) for k, v in kwargs.items() if k != 'save_path'}
    key_names = ["ast_edit_distance", "empty_method",
                 "unreplaced", "pmd_check", "vae"]
    sets_dict = {}
    for key_name in key_names:
        if key_name == "ast_edit_distance":
            sets_dict["$MOC-filter$"] = sets[key_name]
        if key_name == "empty_method":
            sets_dict["$MAD-filter$"] = sets[key_name]
        if key_name == "unreplaced":
            sets_dict["$IM-repairer$"] = sets[key_name]
        if key_name == "pmd_check":
            sets_dict["$BIF-filter$"] = sets[key_name]
        if key_name == "vae":
            sets_dict["$CDF-filter$"] = sets[key_name]

    venn(sets_dict)
    plt.savefig(kwargs["save_path"])
