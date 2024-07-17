import os, datetime
import gseapy as gp
from bioservices.kegg import KEGG
from scripts.args import read_csv
from scripts.utils import make_valid_term


def read_pathway(path: str) -> dict[str, list[str]]:
    df = read_csv(path, index_col=False)
    return {column: df[column].dropna().tolist() for column in df.columns}


def retrieve_pathway(id: str, organism: str) -> dict[str, list[str]]:
    # print('...')
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
    # try:
    #     organism = KEGG_ORGS(organism)
    # except:
    #     raise RuntimeError(f'Organism {organism} is not supported by KEGG annotations yet')

    # k = KEGG()
    # pathway_list = k.list('pathway', organism)
    # pathway_list = pathway_list.split('\n')

    # pathways = {}
    # for pathway in pathway_list:
    #     try:
    #         kegg_id = pathway.split('\t')[0].strip()
    #         pathway_info = k.parse(k.get(kegg_id))
    #         name = pathway_info['NAME'][0].split(' - ')[0]
    #         symbols = [gene.split(';')[0].strip() for gene in pathway_info['GENE'].values()]
    #         pathways[name] = symbols
    #     except:
    #         pass

    # return pathways

    pass


def retrieve_all_msigdb_pathways(organism: str) -> dict[str, list[str]]:
    # assert hs or mm
    raise NotImplementedError('mSigBD is not supported yet')


def get_gene_sets(pathway_database: list[str], custom_pathways: list[str], organism: str) -> dict[str, list[str]]:
    gene_sets = {}

    for database in pathway_database:
        print(f'Retrieving pathways from {database}...')
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
