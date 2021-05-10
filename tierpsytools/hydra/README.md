# Input files required to compile hydra metadata

ATTENTION: All field names should not contain any space (use underscores instead) nor special symbols.

## Manual metadata file:
To be compiled manually during experiment. Contains all the plate-specific information (that are not related to the contents of individual wells) for every experimental run in a specific day.

Name of the file:
    `YYYYMMDD_manual_metadata.csv`

Fields:
1. instrument_name:
    Hydra01, Hydra02, etc (which rig the imaging was)
2. imaging_run_number:
    1, 2, 3, etc (which round of imaging, will roughly map onto datetime of recorded video)
3. imaging_plate_id:
    UNIQUE identifier for a plate. Attention: Must be unique (at least in each imaging day).
    Format when opentrons are used: rr#_sp#_ds#  
        where the three numbers correspond to 'robot_run_id', 'source_plate_id', 'destination_slot' respectively. For example, if robot_run_id=1, source_plate_id=1, destination_slot=4, then the imaging plate id will be 'rr1_sp1_ds4'
    Should be written next to well A1 for manual checking as info gets hardcoded in the videos
4. date_plates_poured_YYYYMMDD:
    date that the imaging plates were poured in

Plus any other imaging-plate-specific metadata you want to record (e.g experimenter, room temperature and humidity).

## Source plate metadata file:
Name of the file:
    `YYYYMMDD_sourceplates.csv`

File written manually. Contains information about the drug contents of every well of every source plate.

Fields:

1. `source_plate_id`:
    Unique identifier for the source plate that holds the drugs to be replicated,
    with or without shuffles, in the imaging plates
2. `well_name`:
    The well name used to define contents
If opentrons are used for shuffling, this file must also contain:
3. `source_robotslot`:
    On which slot, in the robot deck, was said source plate?
4. `robot_runlog_filename`:
    Path to robot runlog (ends in `_runlog.txt`)
[10]. `source_column` or `source_row` or `source_well`: [instead of `well_name`]
    Column/row/well name on the source plate used to define contents

Plus any other metadata you want to record and are related to the source plate contents. 
For example, for drug screenings you need:
`drug_type`: Type of drug in well
`imaging_plate_drug_concentration`: Concentration of the drug in the imaging plate
`imaging_plate_drug_concentration_units`: Units of drug concentration
`stock_drug_concentration`
`solvent` ...
For food-related screenings you need bacteria_strain, ect.
The field name should not contain any space (use underscores instead) nor special symbols.

## Source plate to imgaging plate mapping:
To be compiled manually during experiment. Defines the source plate used to create each imaging plate.

Name of the file:
    `YYYYMMDD_source2imaging.csv`

Only two fields required:
1. `imaging_plate_id`:
    UNIQUE identifier for an imaging plate.
2. `source_plate_id`:
    unique identifier for a source plate, as defined in the source plate metadata file.

Note: This format assumes that the source plate is replicated exactly as is to create the imaging plate. If any shuffling is done between source plate and imaging plate (without the opentron), then this must be taken into account with another file. This option has not been implemented yet.

## Wormsorter file:
To be compiled manually during experiment. Contains well-specific information for every imaging plate used in a given tracking day.
This file should not contain information about the experimental runs (run number, hydra number, time, temperature etc) - this information is recorded in the manual_metadata file.
Instead, it should contain all the basic well-specific metadata for the imaging plates which are not recorded in the sourceplate file: information about the worms put in each well (worm strain, number of worms, days_in_diapause,...) and the media.

Name of the file:
    `YYYYMMDD_wormsorter.csv`

Fields:
1. `imaging_plate_id`:
    UNIQUE identifier for a plate. Attention: Must be unique (at least in each imaging day).
2. `start_well`, `end_well`:
    top left and bottom right well defining a rectangular in the imaging plate. Every well in this rectangular shares the same wormsorter metadata.
3. `media_type`
...

# Compiling metadata

There are two main steps in compiling the metadata:
1. Create complete imaging plate metadata (complete_plate_metadata), which will contain all the information for every well of every imaging plate in a given day.
2. Create the complete metadata for a given tracking day (day_metadata), which will contain all the information for every video recording of every well of every imaging plate in a given day.

See the flowchart for a schematic of the procedure and the functions necessary for different types of experiments.
