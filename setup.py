from distutils.core import setup

setup(
    name="tierpsytools",
    version="0.1dev",
    packages=["tierpsytools"],
    long_description=open("README.md").read(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
                "hydra_sensordata_report=" \
                + "tierpsytools.hydra.read_imgstore_extradata:" \
                + "hydra_sensordata_report",
                "platechecker=" \
                + "tierpsytools.hydra.platechecker:platechecker"
        ]
    },
)
