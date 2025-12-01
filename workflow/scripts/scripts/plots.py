#%%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm



def plot_salt_cavern(regions_path,potentials_path,total_potentail_path,save_path,plot_configs):

    regions = gpd.read_file(regions_path).to_crs(epsg=3035)
    potentials = gpd.read_file(potentials_path).to_crs(epsg=3035).set_index("storage_cluster_type")
    total_potential = gpd.read_file(total_potentail_path).to_crs(epsg=3035)

    #FIXME
    ncols = 2
    nrows = 1 

 
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axs = axs.flatten()

    limit_min = 0
    limit_max = potentials.storage_cap.max()

    cmap = cm.ScalarMappable(cmap='winter_r')
    cmap.set_clim(limit_min, limit_max)

    for i,tt in enumerate(potentials.index.unique()):

        df = potentials.loc[tt]

        # Plot regions in the background
        regions.plot(
            ax=axs[i], 
            facecolor=plot_configs.get("facecolor",'#e6e6e3'), 
            edgecolor=plot_configs.get("edgecolor",'#c2c2c0'), 
            linewidth=plot_configs.get("linewidth",0.5),
            )

        # Plot pipelines for the current cluster
        df.plot(ax=axs[i], column='storage_cap', cmap=plot_configs.get("cmap",'viridis'))

        total_potential.plot(ax=axs[i],markersize=plot_configs.get("markersize",15),color="black")

        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

        
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        
        axs[i].set_title(tt)


    cbar = plt.colorbar(cmap, ax=axs, orientation='vertical', fraction=0.05, pad=0.04, label='Storage Potential ??')
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    
    plt.tight_layout(rect=[0, 0, 0.85, 1.0]) 

    plt.savefig(save_path)




def plot_gas_pipelines(regions_path, pipe_clusters,save_path,plot_configs):
    regions = gpd.read_file(regions_path).to_crs(epsg=3035)
    clusters = gpd.read_file(pipe_clusters).to_crs(epsg=3035).set_index("type")

    unique_clusters = clusters.index.unique()  
    num_clusters = len(unique_clusters)  

    #FIXME
    ncols = 3 
    nrows = 1 

 
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))

 
    axs = axs.flatten()

    clusters['energy_cap'] = pd.to_numeric(clusters['energy_cap'], errors='coerce')
 
    limit_min = 0
    limit_max = clusters['energy_cap'].max() / 1000  # Convert from MW to GW


    cmap = cm.ScalarMappable(cmap=plot_configs.get("cmap",'viridis'))
    cmap.set_clim(limit_min, limit_max)

    for i, tt in enumerate(unique_clusters):
        df = clusters.loc[tt]

        # Plot regions in the background
        regions.plot(
            ax=axs[i], 
            facecolor=plot_configs.get("facecolor",'#e6e6e3'), 
            edgecolor=plot_configs.get("edgecolor",'#c2c2c0'), 
            linewidth=plot_configs.get("linewidth",0.5),
            )

        # Plot pipelines for the current cluster
        df.plot(ax=axs[i], column='energy_cap', cmap=plot_configs.get("cmap",'viridis'))

        
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

        
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        
        axs[i].set_title(tt)

    
    cbar = plt.colorbar(cmap, ax=axs, orientation='vertical', fraction=0.05, pad=0.04, label='Capacity ??')
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    
    plt.tight_layout(rect=[0, 0, 0.85, 1.0]) 

    plt.savefig(save_path)
    
    return fig


def plot(
        regions_path,
        potentials_path,
        total_potentail_path,
        salt_cavern_save_path,
        pipe_cluster_path,
        pipe_cluster_save_path,
        plot_configs,

):
    plot_salt_cavern(
        regions_path=regions_path,
        potentials_path=potentials_path,
        total_potentail_path=total_potentail_path,
        save_path=salt_cavern_save_path,
        plot_configs = plot_configs,
    )

    plot_gas_pipelines(
        regions_path=regions_path,
        pipe_clusters=pipe_cluster_path,
        save_path=pipe_cluster_save_path,
        plot_configs = plot_configs,
    )


#%%
if __name__ == "__main__":
    plot(
        regions_path=snakemake.input.regions,
        potentials_path=snakemake.input.salt_cavern,
        total_potentail_path=snakemake.input.salt_cavern_potential,
        salt_cavern_save_path=snakemake.output.salt_cavern,
        pipe_cluster_path=snakemake.input.pipe_clusters,
        pipe_cluster_save_path=snakemake.output.pipe_clusters,
        plot_configs=snakemake.params.plot_configs
    )





# %%
