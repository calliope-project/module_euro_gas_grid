"""Rules to standardise files."""


rule prepare_landmass:
    message:
        "Preparing landmass data."
    input:
        raw_folder=rules.unzip_landmass.output.folder,
    output:
        landmass="resources/automatic/landmass.parquet",
        fig="resources/automatic/landmass.png"
    log:
        "logs/prepare_landmass.log",
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_landmass.py"



rule prepare_pipelines:
    message:
        "Validating SciGRID data."
    input:
        raw_pipelines=rules.unzip_pipe_segements.output.pipelines,
        landmass=rules.prepare_landmass.output.landmass
    output:
        pipelines="resources/automatic/pipelines.parquet",
        fig="resources/automatic/pipelines.png"
    log:
        "logs/prepare_pipelines.log",
    conda:
        "../envs/euro_gas_grid.yaml"
    script:
        "../scripts/prepare_pipelines.py"
