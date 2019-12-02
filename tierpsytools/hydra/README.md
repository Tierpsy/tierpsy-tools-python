## Source plate metadata file:

Name of the file:
    `YYYYMMDD_sourceplates.csv`

File written manually (before and after a robot run). Contains all the information necessary to compile the metadata for the plates created with the robot.

Fields:

1. `source_plate_id`:
    Unique identifier for the plate that holds the drugs to be replicated,
    with shuffles, in the imaging plates
2. `source_robotslot`:
    On which slot, in the robot deck, was said source plate?
3. `robot_runlog_filename`:
    Path to robot runlog (ends in `_runlog.txt`)
4. `source_column` or `source_row` or `source_well`:
    Column/row/well name on the source plate used to define contents

Plus any other metadata you want to record and are related to the source plate contents. 
For example, for drug screenings you need drug_type, drug_dose, drug_dose_units. For food-related screenings you need bacteria_strain, ect.
Other recommended fields are number_worms_per_well, worm_strain, media_type,...
The field name should not contain any space (use underscores instead) nor special symbols.

## Manual metadata file:
To be compiled manually during experiment. Contains all the plate-specific information about the experiment (that are not related to the contents of individual wells).

Name of the file:
    `YYYYMMDD_manual_metadata.csv`

Fields:
1. instrument_name:
    Hydra01, Hydra02, etc (which rig the imaging was)
2. imaging_run_number:
    1, 2, 3, etc (which round of imaging, will roughly map onto datetime of recorded video)
3. imaging_plate_id:
    Unique (at least in each imaging day) identifier for a plate.
    Format: rr#_sp#_ds#  
        where the three numbers correspond to 'robot_run_id', 'source_plate_id', 'destination_slot' respectively. For example, if robot_run_id=1, source_plate_id=1, destination_slot=4, then the imaging plate id will be 'rr1_sp1_ds4'
    Should be written next to well A1 for manual checking as info gets hardcoded in the videos
4. date_plates_poured_YYYYMMDD:
    date that the imaging plates were poured in 

Plus any other imaging-plate-specific metadata you want to record (e.g experimenter, room temperature and humidity).
The field name should not contain any space (use underscores instead) nor special symbols.


## Automatic metadata file:

To be created automatically by tierpsytools script that will combine:
    `YYYYMMDD_sourceplates`
    `YYYYMMDD_manual_metadata`
    robot runlog

Name of the file:
    `YYYYMMDD_day_metadata.csv`

Fields:
1. date_yyyymmdd:
    date of imaging
2. imaging_run_number
3. instrument_name
4. imaging_plate_id
5. imaging_well_name
6. source_robotslot
7. source_plate_id
8. source_column
9. source_well
10. transferred_volume:
    volume transferred from source well to well in imaging plate
11. destination_robotslot
12. robot_runlog_filename
13. date_plates_poured_YYYYMMDD
