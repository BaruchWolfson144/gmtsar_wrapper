from make_dir_tree_01 import run_create_project
from make_dem_02 import run_make_dem
from make_orbits_04 import run_download_orbits
from choose_master_05 import run_preprocess_subswath
from run_interferograms_06 import run_intf
from merge_intfs_07 import run_merge
from unwrap_intfs_08 import run_unwrap

    

"""
make meta_config file that user will fill with all his wish (like polygon cordinates) and then run all scripts acoordings to data in this config file
"""


subswath_list = ["F1", "F2", "F3"]
print(run_create_project(project_root, desc)) #01
print(run_make_dem(project_root, bbox, mode, make_dem_path)) #02
print(run_download_orbits(project_root, mode, stitch)) #04
for sub in subswath_list:
    print(run_preprocess_subswath(project_root, orbit, sub)) #05
    print (run_intf(project_root, orbit, sub, threshold_time, threshold_baseline, config_path, master) #06
print(run_merge(project_root, orbit, master, mode)) #07 # user need give mode to choose wich subs merge (defult: all)
)

