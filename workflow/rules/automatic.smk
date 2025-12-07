"""Rules to used to download automatic resource files."""

wildcard_constraints:
    nat_earth="|".join(["landmass", "countries"]),
    scigrid_gas="|".join(["BorderPoints", "Compressors", "Consumers", "LNGs", "Nodes", "PipeSegments", "PowerPlants", "Productions", "Storages"])

rule download_sci_grid:
    message:
        "Downloading gas infrastructure data from SciGRID_gas IGGIELGN."
    params:
        url = internal["resources"]["automatic"]["scigrid_gas"]
    log:
        "logs/automatic/download_sci_grid.log",
    output:
        zipfile="resources/automatic/gas_grid.zip",
    localrule: True,
    conda:
        "../envs/shell.yaml"
    shell:
        """curl -sSLo {output} {params.url}"""


rule unzip_scigrid_dataset:
    message:
        "Unzipping SciGrid '{wildcards.scigrid_gas}'."
    params:
        file=lambda wc: f"data/IGGIELGNC3_{wc.scigrid_gas}.geojson",
    input:
        zip_file=rules.download_sci_grid.output.zipfile,
    output:
        pipelines="resources/automatic/scigrid_gas/{scigrid_gas}.geojson",
    log:
        "logs/automatic/unzip_scigrid_dataset_{scigrid_gas}.log",
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
        "logs/automatic/download_{nat_earth}.log",
    output:
        zipfile="resources/automatic/{nat_earth}.zip",
    localrule: True,
    conda:
        "../envs/shell.yaml"
    shell:
        """curl -sSLo {output} {params.url}"""


rule download_north_sea:
    message:
        "Downloading North Sea map from the marine regions database."
    params:
        url = internal["resources"]["automatic"]["marine_regions"]
    log:
        "logs/automatic/download_north_sea.log"
    output:
        zipfile="resources/automatic/north_sea_2350.zip",
    localrule: True,
    conda:
        "../envs/shell.yaml"
    shell:
        """
        curl -sSL -G {params.url:q} \
        --data-urlencode 'service=WFS' \
        --data-urlencode 'version=2.0.0' \
        --data-urlencode 'request=GetFeature' \
        --data-urlencode 'typeNames=iho' \
        --data-urlencode 'cql_filter=mrgid=2350' \
        --data-urlencode 'outputFormat=SHAPE-ZIP' \
        -o {output:q}
        """

