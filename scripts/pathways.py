import os, datetime
import pandas as pd
import gseapy as gp
from scripts.args import read_csv
from scripts.utils import make_valid_term, make_valid_filename


def read_pathway(path: str) -> dict[str, list[str]]:
    df = read_csv(path, index_col=False)
    return {column: df[column].dropna().tolist() for column in df.columns}


def retrieve_pathway(id: str, organism: str) -> dict[str, list[str]]:
    raise NotImplementedError('Pathway ID is not supported yet')


def retrieve_all_go_pathways(organism: str, pathway_type: str = None) -> dict[str, list[str]]:
    
    def get_go_db(db):
        if db.lower().replace('_', ' ') in ['molecular function', 'mf']:
            return 'GO_Molecular_Function'
        elif db.lower().replace('_', ' ') in ['biological process', 'bp']:
            return 'GO_Biological_Process'
        elif db.lower().replace('_', ' ') in ['cellular component', 'cc']:
            return 'GO_Cellular_Component'
        raise ValueError(f'Invalid GO database {db}')

    def get_library(db, organism):
        db = get_go_db(db)
        all_libraries = gp.get_library_name(organism=organism)

        curr_year = datetime.datetime.now().year
        for year in range(curr_year, 2018, -1):
            newest_db = [lib for lib in all_libraries if f'{db}_{year}' == lib]
            if newest_db:
                return newest_db[0]
        raise RuntimeError('No available GO databsae')
     
    def get_go_pathways(db, organism):
        library = get_library(db, organism)
        pathways = gp.get_library(library, organism=organism)
        return pathways

    pathway_types = [pathway_type] if pathway_type else ['bp', 'mf', 'cc']
    pathways = {}
    for pathway_type in pathway_types:
        pathways.update(get_go_pathways(pathway_type, organism))

    pathways = {key.split(' (GO')[0]: value for key, value in pathways.items()}
    return pathways


def retrieve_all_kegg_pathways(organism: str) -> dict[str, list[str]]:
    raise NotImplementedError('KEGG is not supported yet')


def retrieve_all_msigdb_pathways(organism: str) -> dict[str, list[str]]:
    # assert hs or mm
    raise NotImplementedError('mSigBD is not supported yet')


def get_gene_sets(pathway_database: list[str], custom_pathways: list[str], organism: str) -> dict[str, list[str]]:
    gene_sets = {}

    for database in pathway_database:
        retrieval = globals()[f'retrieve_all_{database}_pathways']
        gene_sets.update(retrieval(organism))

    for pathway in custom_pathways: 
        # Read gene sets
        if os.path.exists(pathway):
            gene_sets.update(read_pathway(pathway))
        # Retrieve gene set from database by ID
        else:
            gene_sets.update(retrieve_pathway(pathway, organism))
    
    gene_sets = {make_valid_term(key): value for key, value in gene_sets.items()}
    return gene_sets
    
        
def export_gene_sets(gene_sets: dict[str, list[str]], out_path: str, by_set: bool = True) -> None:
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in gene_sets.items()]))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    df.to_csv(f'{out_path}/gene_sets.csv', index=False)
    if by_set:
        for col in df.columns:
            pd.DataFrame(df[col]).dropna().to_csv(f'{out_path}/{make_valid_filename(col)}.csv', index=False)
