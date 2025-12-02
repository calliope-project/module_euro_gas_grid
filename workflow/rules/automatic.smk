"""Rules to used to download automatic resource files."""

wildcard_constraints:
    nat_earth="|".join(["landmass", "countries"])


rule download_sci_grid:
    message:
        "Downloading gas infrastructure data from SciGRID_gas IGGIELGN."
    params:
        url = internal["resources"]["automatic"]["scigrid_gas"]
    log:
        "logs/download_sci_grid.log",
    output:
        zipfile=temp("resources/automatic/gas_grid.zip"),
    localrule: True,
    conda:
        "../envs/shell.yaml"
    shell:
        """curl -sSLo {output} {params.url}"""


rule unzip_pipe_segements:
    message:
        "Unzipping SciGrid '{params.file}'."
    params:
        file=f"data/IGGIELGN_PipeSegments.geojson",
    input:
        zip_file=rules.download_sci_grid.output.zipfile,
    output:
        pipelines=temp("resources/automatic/pipesegments.geojson"),
    log:
        "logs/automatic/unzip_pipe_segements.log",
    conda:
        "../envs/clustering.yaml"
    script:
        "../scripts/unzip.py"


rule download_natural_earth:
    message:
        "Downloading '{wildcards.nat_earth}' from Natural Earth data (10m)."
    params:
        url = lambda wc: internal["resources"]["automatic"]["natural_earth"][wc.nat_earth]
    log:
        "logs/download_{nat_earth}.log",
    output:
        zipfile=temp("resources/automatic/{nat_earth}.zip"),
    localrule: True,
    conda:
        "../envs/shell.yaml"
    shell:
        """curl -sSLo {output} {params.url}"""


rule unzip_natural_earth:
    message:
        "Unzipping natural earth '{wildcards.nat_earth}' data."
    input:
        zip_file=rules.download_natural_earth.output.zipfile,
    output:
        folder=directory("resources/automatic/{nat_earth}/"),
    log:
        "logs/automatic/unzip_natural_earth_{nat_earth}.log",
    conda:
        "../envs/clustering.yaml"
    script:
        "../scripts/unzip.py"
