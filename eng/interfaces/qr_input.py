import re, json, requests, time, os
from typing import List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import io
from IPython.display import display, Markdown
import ase.io
from ase import Atoms
from ase.spacegroup.symmetrize import check_symmetry

from ..assets.assets_collection import pp_json_path,sg_dict, mc3d_db, mc2d_db
from ..assets.text_prompt import markdown_table

def parse_chemical_formula(formula: str) -> dict:
    """
    Parses a chemical formula string into a dictionary of element symbols and their counts.

    Args:
        formula: The chemical formula string (e.g., "H2O", "NaCl", "Fe2O3").

    Returns:
        A dictionary where keys are element symbols (e.g., "H", "Na", "Fe") and values
        are their corresponding counts as integers. If an element has no explicit count
        in the formula, its count is assumed to be 1.
    """
    # Regular expression to match element symbols and their counts
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)

    element_counts = {}
    for element, count in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        element_counts[element] = count

    return element_counts


#def dict_str(d) -> str:
#    # foramt the dictionary as a string, keys as headers and values as content
#    if isinstance(d, str):
#        return d
#    out = ""
#    for k, v in d.items():
#        if isinstance(v, dict):
#            out += f"<{k}>\n{dict_str(v)}\n</{k}>\n\n"
#        else:
#            out += f"<{k}>\n{v}\n</{k}>\n\n"
#    return out



def pseudo_file(element_name : str, json_path: str = pp_json_path) -> str:
    """returns the pseudo_potential file, cutoff_wfc, and cutoff_rho for the given element name."""
    with open(json_path) as file:
        pseudo_files = json.load(file)
    dic = pseudo_files[element_name]
    dic['pseudo_potential_name'] = dic['filename']
    dic['ecutwfc'] = dic['cutoff_wfc']
    dic['ecutrho'] = dic['cutoff_rho']
    del dic['cutoff_wfc']
    del dic['cutoff_rho']
    del dic['md5']
    del dic['pseudopotential']
    del dic['filename']
    return dic #dict_str(dic)


def extract_pseudo_from_text(text: str) -> str | None:
    """
    Extracts the pseudopotential filename from a string containing ATOMIC_SPECIES information.

    Args:
        text: The input string containing ATOMIC_SPECIES line.

    Returns:
        The pseudopotential filename if found, otherwise None.
    """

    sliced_text = text[text.find('ATOMIC_SPECIES'):text.find('ATOMIC_POSITIONS')]

    basic_pattern = r'^\s*\w+\s+[\d.]+\s+(\S+)'

    matches = re.findall(basic_pattern, sliced_text, re.MULTILINE)
    if matches:
        return matches
    return None

def get_pseudo_file(pp_file: str, element: str, pp_dir: Path) -> Path | list[Path] | None:
    """
    Retrieves pseudopotential file(s) from a directory.
    This function searches for pseudopotential files based on either a given filename or an element symbol.
    If a filename is provided, it checks for the existence of that specific file. If an element symbol is provided,
    it searches for files matching the element name pattern.

    Args:
        pp_file (str): The specific pseudopotential filename to search for. If None or empty, the function will search by element.
        element (str): The element symbol to search for (e.g., 'Fe'). This is used if pp_file is not provided.
        pp_dir (Path): The directory containing the pseudopotential files.

    Returns:
        Path | list[Path] | None:
            - If pp_file is given and a matching file exists, returns the Path to that file.
            - If pp_file is not given, returns a list of Path objects for all matching files based on the element.
            - Returns None if pp_file is given but not found.

    Raises:
        FileNotFoundError: If no matching file is found for the given element when pp_file is not provided.
    """
    if pp_file:
        pp_file_path = Path(pp_dir, pp_file)
        if pp_file_path.exists() and pp_file_path.is_file():
            return pp_file_path
        else:
            return None
    element = element.lower()
    pattern = rf"^{element}[.\-_]"

    matching_files = [file for file in pp_dir.glob('*') if re.match(pattern, file.name.lower())]
    
    if not matching_files:
        raise FileNotFoundError(f"No pseudopotential file found for element {element}")

    return matching_files

#def input_parameters(chemical_formula: str) -> dict:
#    """Retrieve the CELL_PARAMETERS, ATOMIC_POSITIONS and Pseudo potential file."""
#    global folder_manager
#    compound = parse_chemical_formula(chemical_formula)
#    uuidds = []
#    uuidd2s = []
#    URL = f"https://www.materialscloud.org/mcloud/api/v2/discover/mc3d/compounds/{chemical_formula}"
#    response = requests.get(URL)
#    uuidds = []
#    if response.status_code == 200:
#        col = response.json()
#        for data in col['data']:
#            if data['source'][0]['info']['is_high_pressure'] or data['source'][0]['info']['is_high_temperature']:
#                continue
#
#            items = data['provenance_links']
#            
#            for item in items:
#                if item['label'] == 'Final SCF calculation':
#                    uuidds.append(item['uuid'])
#                if item['label'] == 'Final structure':
#                    uuidd2s.append(item['uuid'])
#
#    else:
#        return 'try again with correctly capitalized formula, or reverse the order of element, example: wrong: feo, correct FeO, wrong: ZnO, correct: OZn'
#    
#    output_dict_list = []
#    for uuidd2 in uuidd2s:
#        url2 = f"https://aiida.materialscloud.org/mc3d/api/v4/nodes/{uuidd2}/download?download_format=xyz"
#        response2 = requests.get(url2)
#        
#        atomic_position = ''
#        if response2.status_code == 200:
#            with open("tmp.xyz", "wb") as file:
#                file.write(response2.content)
#            c_cell = ase.io.read('tmp.xyz')
#            os.remove('tmp.xyz')
#            out_cell = c_cell.cell.array.round(3)
#            out_cell_st = ''
#            for i in out_cell:
#                out_cell_st += '\t'.join([str(k) for k in i])
#                out_cell_st += '\n'
#            
#            cs = c_cell.get_chemical_symbols()
#            p =  c_cell.get_positions().tolist()
#            x = list(zip(cs,p))
#            t = ''
#            for i in x:
#                t += i[0] + '\t'
#                for j in i[1]:
#
#                    t +=str(round(j+1e-5,4))
#                    t+= '\t'
#                t += '\n'
#            atomic_position = re.sub(f'\t\n', '\n', t)
#        p_file = pseudo_file.batch([{'element_name' : elm, 'json_path' : pp_json_path,
#
#                                } for elm in compound.keys()])
#        output_dict = {"cell_parameters_angstrom" : out_cell, "atomic_positions_angstrom" : atomic_position,'raw': x, 'pseudo_file': p_file}
#        output_dict_list.append(output_dict)
#    
#    return output_dict_list


#def tot_maker(compound, chem_formula):
#    p_file = pseudo_file.batch([{'element_name' : elm, 'json_path' : pp_json_path,
#
#                                } for elm in compound.keys()])
#    
#    inp_data = input_parameters(chemical_formula = chem_formula )
#
#    cell = ""
#    for ii in inp_data['cell_parameters_angstrom'].tolist():
#        for jj in ii:
#            cell += str(jj) + '\t'
#        cell += '\n'
#
#    tot = "" #"""<pseudo_dir>\n/home/ws/ec5456/VS-nano/langc/matsim/new_pp\n</pseudo_dir>\n<outdir>\n/home/ws/ec5456/VS-nano/langc/matsim/out_dir\n</outdir>\n"""
#    for ii in p_file:
#
#        tot += "<ELEMENT>\n"
#        tot += ii
#        #tot += '\n'
#        tot += '</ELEMENT>\n'
#        
#    tot += "CELL_PARAMETERS angstrom\n"
#    tot += cell
#    tot += '\n'
#    tot += "ATOMIC_POSITIONS angstrom\n"
#    tot += inp_data['atomic_positions_angstrom']
#
#    return tot.replace('\n\n', '\n')
#
def find_compound(chemical_formula: str, 
                mc3d_db: dict = mc3d_db,
                stoichiometric:bool = False ) -> list[list, dict]|str:
    
    """Find the compound with the given chemical formula."""
    compound = parse_chemical_formula(chemical_formula)
    # load json file
    string = list(compound.keys())
    coll_chem = []
    for k, v in mc3d_db.items():
        if  'flg' in v.keys():
            continue

        db_cmp = list(parse_chemical_formula(v['formula']).keys())
        inclusion = all([s in db_cmp for s in string])
        if not inclusion:
            continue
        
        x_a = parse_chemical_formula(v['formula'])
        string2 = list(x_a.keys())

        main_1 = [s in string for s in string2]
        
        if not all(main_1):
            continue
        
        if stoichiometric:
            x_a_n = np.array([x_a.get(s) for s in string2])
            gcd_a = np.gcd.reduce(x_a_n)
            x_b = np.array([compound.get(s) for s in string2])
            gcd_b = np.gcd.reduce(x_b)
            stoch_match = (x_a_n / gcd_a == x_b / gcd_b).all()
            #print(v['formula'], string2, string, x_a, compound, x_a_n, gcd_a, x_b, gcd_b, stoch_match)
            if not stoch_match:
                continue
            else:
                coll_chem.append(v)

        else:
            if string2 == string or all(main_1):
                coll_chem.append(v)
            

            
    if len(coll_chem) == 0:
        return 'No compound found with the given chemical formula.'
    else:
        compound = [i['formula'] for i in coll_chem]
        return [coll_chem, compound]
    

def qe_input_query(qe_chem_formula: str) -> list[tuple[str, str]]:
    """
    Retrieves Quantum Espresso input files from Materials Cloud for a given chemical formula.

    This function queries the Materials Cloud database for structures matching the provided
    chemical formula and extracts the input files used for the final self-consistent field (SCF)
    calculation. It specifically filters out high-pressure and high-temperature structures.

    Args:
        qe_chem_formula (str): The chemical formula of the material (e.g., "Al", "SiO2", "OZn", "Fe2O3").

    Returns:
        list[tuple[str, str]]: A list of tuples, where each tuple contains:
            - str: The content of the Quantum Espresso input file with the prefix modified to 'qe_{compound}'.
            - str: The UUID of the AiiDA node from which the input file was retrieved.
        Returns 'try again...' if the request fails.
    """
    compound = qe_chem_formula
    URL = f"https://www.materialscloud.org/mcloud/api/v2/discover/mc3d/compounds/{compound}"
    response = requests.get(URL)
    uuidds = []
    if response.status_code == 200:
        col = response.json()
        for data in col['data']:
            if data['source'][0]['info']['is_high_pressure'] or data['source'][0]['info']['is_high_temperature']:
                continue

            items = data['provenance_links']
            
            for item in items:
                if item['label'] == 'Final SCF calculation':
                    uuidd = item['uuid']
                    uuidds.append(uuidd)

    else:
        return 'try again...'
    
    out_txt = []
    for uuidd in uuidds:
        url = f"https://aiida.materialscloud.org/mc3d/api/v4/nodes/{uuidd}/repo/contents?filename=%22aiida.in%22"
        response = requests.get(url)
        if response.status_code == 200:
            #with open(f"qe_{compound}.in", "wb") as file:
            print("File downloaded successfully.")
            x_ = response.content.decode('utf-8').replace("prefix = 'aiida'",f"prefix = 'qe_{compound}'" )
            out_txt.append([x_, uuidd])
        
        
        else:
            continue
    return out_txt

    
def create_atoms_from_strings(atomic_positions_str, cell_parameters_str):
    # Parse atomic positions
    lines = atomic_positions_str.strip().split('\n')[1:]  # Skip the header
    symbols = []
    positions = []
    for line in lines:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(p) for p in parts[1:4]])

    # Parse cell parameters
    cell_lines = cell_parameters_str.strip().split('\n')[1:]  # Skip the header
    cell = []
    for line in cell_lines:
        cell.append([float(p) for p in line.split()])

    # Create the Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
    atoms.set_pbc([True, True, True])  # Set periodic boundary conditions

    return atoms


#def present_find_out(find_out: dict) -> str:
#    #check type of find_out
#    if isinstance(find_out, str):
#        return find_out
#
#    tx_ = pd.json_normalize(find_out)
#    tx_['sg'] = tx_['sg'].apply(lambda x: sg_dict.get(str(x)))
#    #xc.pbe.am	xc.pbe.tm
#    tx_.rename(columns={'sg': 'Space Group', 'xc.pbe.am': 'xc.pbe: abs. mag.',
#                        'formula': 'Formula',
#                        'xc.pbe.tm': 'xc.pbe: tot. mag.'}, inplace=True)
#    # start index from 1
#    tx_.index = tx_.index 
#    tx__ = tx_.to_markdown()
#    
#    display(Markdown(markdown_table.format(text = tx__)))
#    return tx_

#def helper_retriever(find_out: list, pp_dir: Path) -> list:
#    """
#    Helper function to retrieve and format Quantum Espresso input files for a list of compounds.
#    
#    Args:
#        find_out: List of dictionaries containing compound information
#        pp_dir: Path to the directory containing pseudopotential files
#        
#    Returns:
#        List of tuples containing Atoms objects, space group numbers, and formatted Quantum Espresso input files
#    """
#    x_sg = []
#    chem_data, chem_formulas = find_out
#
#    for chem_formula in chem_formulas:
#        x_ = qe_input_query(qe_chem_formula= chem_formula)
#        for qe_inp in x_:
#            atomic_positions = []
#            cell_parameters = []
#            cards = re.findall(r'\n[A-Z_]{2,}', qe_inp, re.DOTALL)
#            cards = [i.strip() for i in cards]
#            compx = parse_chemical_formula(chem_formula)
#            elements = list(compx.keys())
#
#            original_pp_files = extract_pseudo_from_text(qe_inp)
#            #available_pp_files = []
#            for pp_file in original_pp_files:
#                pp_files_real = get_pseudo_file(pp_file=pp_file, element='', pp_dir=pp_dir)
#                if isinstance(pp_files_real, Path):
#                    pass
#
#                else:
#                    element = re.split(r'[._-]', pp_file)[0]
#                    pp_files_real = get_pseudo_file(pp_file='', element= element, pp_dir=pp_dir)[0]
#                    # get the pp_file name
#                    pp_file_name = pp_files_real.name
#                    # replace the pp_file name in the qe_inp
#                    qe_inp = re.sub(pp_file, pp_file_name, qe_inp)
#
#
#            carde_st = ''
#            ecutwfc , ecutrho = 0,0
#            for element in elements:
#                pseudo = pseudo_file(element_name = element, json_path = pp_json_path)
#                if pseudo['ecutwfc'] > ecutwfc:
#                    ecutwfc = pseudo['ecutwfc']
#                if pseudo['ecutrho'] > ecutrho:
#                    ecutrho = pseudo['ecutrho']
#
#            carde_st += f'&SYSTEM ecutwfc = {pseudo["ecutwfc"]}\n'
#            carde_st += f'&SYSTEM ecutrho = {pseudo["ecutrho"]}\n'
#            carde_st += '&SYSTEM ibrav = 0' + '\n'
#            carde_st += f'&SYSTEM nat = {sum(list(compx.values()))}' + '\n' 
#            carde_st += '-'*10 + '\n'
#            for card in cards:
#                card_item = re.findall(rf'(?={card}).*?(?=\n[A-Z_]{{2,}}|$)', qe_inp, re.DOTALL)[0]
#                carde_st += card_item + '\n'
#                if 'CELL_PARAMETERS' in card:
#                    cell_parameters.append(card_item)
#                if 'ATOMIC_POSITIONS' in card:
#                    atomic_positions.append(card_item)
#                carde_st += '\n'
#            try:
#                weird_name = re.findall(r'[A-Z][a-z]?\d{1,2}', atomic_positions[0], re.DOTALL)
#                for i in weird_name:
#                    A_Z = re.findall(r'[A-Z][a-z]?', i, re.DOTALL)[0]
#                    digit = re.findall(r'\d{1,2}', i, re.DOTALL)[0]
#                    atomic_positions[0] = re.sub(i, f'{A_Z}', atomic_positions[0], re.DOTALL)
#            except:
#                pass
#
#            atoms = create_atoms_from_strings(atomic_positions[0], cell_parameters[0])
#            symmetry = check_symmetry(atoms, symprec=1e-3)
#
#            x_sg.append([ atoms, symmetry.number, sg_dict.get(str(symmetry.number)), qe_inp, carde_st])
#
#    return x_sg

def qe_polisher(formula: str, qe_inp: str, pp_dir: Path) -> tuple[Atoms, int, str, str, str]:
    """
    Processes a Quantum Espresso input file, extracts structure information, and returns relevant data.

    This function takes a chemical formula, a Quantum Espresso input string, and a path to pseudopotential files.
    It parses the input, corrects pseudopotential file names, extracts atomic positions and cell parameters,
    determines the space group, and returns the processed data.

    Args:
        formula (str): The chemical formula of the material.
        qe_inp (str): The Quantum Espresso input file content as a string.
        pp_dir (Path): The path to the directory containing pseudopotential files.

    Returns:
        list: A list containing:
            - atoms (ase.Atoms): An ASE Atoms object representing the structure.
            - symmetry.number (int): The space group number.
            - sg_dict.get(str(symmetry.number)) (str): The space group symbol.
            - qe_inp (str): The modified Quantum Espresso input string with corrected pseudopotential names.
            - carde_st (str): A string containing the processed card information.
    """

    original_pp_files = extract_pseudo_from_text(qe_inp)
    for pp_file in original_pp_files:
        pp_files_real = get_pseudo_file(pp_file=pp_file, element='', pp_dir=pp_dir)
        if isinstance(pp_files_real, Path):
            pass

        else:
            element = re.split(r'[._-]', pp_file)[0]
            pp_files_real = get_pseudo_file(pp_file='', element= element, pp_dir=pp_dir)[0]
            # get the pp_file name
            pp_file_name = pp_files_real.name
            # replace the pp_file name in the qe_inp
            qe_inp = re.sub(pp_file, pp_file_name, qe_inp)
    
    atomic_positions = []
    cell_parameters = []
    cards = re.findall(r'\n[A-Z_]{2,}', qe_inp, re.DOTALL)
    cards = [i.strip() for i in cards]
    compx = parse_chemical_formula(formula)
    carde_st = ''
    ecutwfc , ecutrho = 0,0
    elements = list(compx.keys())

    for element in elements:
        pseudo = pseudo_file(element_name = element, json_path = pp_json_path)
        if pseudo['ecutwfc'] > ecutwfc:
            ecutwfc = pseudo['ecutwfc']
        if pseudo['ecutrho'] > ecutrho:
            ecutrho = pseudo['ecutrho']

    carde_st += f'&SYSTEM ecutwfc = {pseudo["ecutwfc"]}\n'
    carde_st += f'&SYSTEM ecutrho = {pseudo["ecutrho"]}\n'
    carde_st += '&SYSTEM ibrav = 0' + '\n'
    carde_st += f'&SYSTEM nat = {sum(list(compx.values()))}' + '\n' 
    carde_st += '-'*10 + '\n'
    for card in cards:
        card_item = re.findall(rf'(?={card}).*?(?=\n[A-Z_]{{2,}}|$)', qe_inp, re.DOTALL)[0]
        carde_st += card_item + '\n'
        if 'CELL_PARAMETERS' in card:
            cell_parameters.append(card_item)
        if 'ATOMIC_POSITIONS' in card:
            atomic_positions.append(card_item)
        carde_st += '\n'
    try:
        weird_name = re.findall(r'[A-Z][a-z]?\d{1,2}', atomic_positions[0], re.DOTALL)
        for i in weird_name:
            A_Z = re.findall(r'[A-Z][a-z]?', i, re.DOTALL)[0]
            atomic_positions[0] = re.sub(i, f'{A_Z}', atomic_positions[0], re.DOTALL)
    except:
        pass

    atoms = create_atoms_from_strings(atomic_positions[0], cell_parameters[0])
    symmetry = check_symmetry(atoms, symprec=1e-3)

    return [ atoms, symmetry.number, sg_dict.get(str(symmetry.number)), qe_inp, carde_st]


class HelperRetrieverMc2d:
    """
    A class to handle retrieval and formatting of material structure data from MaterialsCloud.
    """
    def get_uuid(formula: str) -> str:
        """
        Get the UUID of the structure for a given formula
        """
        uuid = mc2d_db['compounds'][formula][0]['structure_2D']
        return uuid
    
    @staticmethod
    def download_cif_io(uuid: str) -> io.BytesIO:
        """
        Download CIF file for a given UUID from MaterialsCloud.
        
        Args:
            uuid: The UUID of the structure
            
        Returns:
            str: Content of the CIF file
        """
        url = f"https://aiida.materialscloud.org/mc2d/api/v4/nodes/{uuid}/download?download_format=cif"
        response = requests.get(url)
        return response.content
    
    @staticmethod
    def uuid_atoms(uuid: str) -> Atoms:
        """
        Download and save CIF file for a given UUID.
        
        Args:
            uuid: The UUID of the structure
            output_path: Path where to save the CIF file
        """
        cif_content = HelperRetrieverMc2d.download_cif_io(uuid)
        
        cif_content_io = io.BytesIO(cif_content)
        atoms = ase.io.read(cif_content_io, format='cif')
        return atoms
    
    @staticmethod
    def format_qe_structure(atoms: Atoms, 
                          k_points_2d: Optional[List[int]], 
                          pseudo_dir: Path ) -> str:
        """
        Convert ASE atoms object data into formatted Quantum Espresso structure input.
        
        Args:
            atoms: ASE atoms object containing structure information
            k_points: List of 6 integers [nk1, nk2, nk3, sk1, sk2, sk3] for k-point mesh
            pseudo_dir: Path to directory containing pseudopotential files
            
        Returns:
            str: Formatted Quantum Espresso structure input text
            
        Raises:
            ValueError: If k_points length is incorrect, pseudo_dir is None or doesn't exist,
                      or if pseudopotential files are not found
        """
        # set initial k_points
        if k_points_2d is None:
            k_points_2d = [7, 7, 2, 0, 0, 0]

        # Input validation
        if len(k_points_2d) != 6:
            raise ValueError("k_points_2d must be a list of 6 integers")
        
        if pseudo_dir is None:
            raise ValueError("pseudo_dir must be specified")
            
        if not pseudo_dir.exists():
            raise ValueError(f"Pseudopotential directory {pseudo_dir} does not exist")
        
        # Get required data
        symbols = atoms.get_chemical_symbols()
        masses = atoms.get_masses()
        positions = atoms.positions
        cell = atoms.cell.todict()['array']
        
        # Get unique elements and their masses
        unique_elements = list(dict.fromkeys(symbols))
        species_text = []
        
        for element in unique_elements:
            # Find pseudopotential file for this element
            try:
                pp_files = get_pseudo_file(pp_file=None, element=element, pp_dir=pseudo_dir)
                if isinstance(pp_files, list):
                    pp_file = pp_files[0].name
                else:
                    raise FileNotFoundError(f"No pseudopotential file found for element {element} in {pseudo_dir}")
            except FileNotFoundError as e:
                raise ValueError(str(e))
            
            # Use the first matching pseudopotential file
            pp_file = pp_files[0].name
            mass = masses[symbols.index(element)]
            species_text.append(f"{element:4} {mass:8.3f} {pp_file}")

        ecutwfc , ecutrho = 0,0
        for element in unique_elements:
            pseudo = pseudo_file(element_name = element, json_path = pp_json_path)
            if pseudo['ecutwfc'] > ecutwfc:
                ecutwfc = pseudo['ecutwfc']
            if pseudo['ecutrho'] > ecutrho:
                ecutrho = pseudo['ecutrho']

        
        # Format sections
        sections = [
            (f"&SYSTEM\n  ecutwfc = {ecutwfc}\n  ecutrho = {ecutrho}\n  ibrav = 0\n  nat = {len(atoms)}\n/\n"),

            ("ATOMIC_SPECIES\n" + "\n".join(species_text)),
            "",
            ("ATOMIC_POSITIONS angstrom\n" + 
             "\n".join(f"{s:4} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}" 
                      for s, p in zip(symbols, positions))),
            "",
            "K_POINTS automatic\n" + " ".join(str(k) for k in k_points_2d),
            "",
            ("CELL_PARAMETERS angstrom\n" + 
             "\n".join(f"{v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}" 
                      for v in cell))
        ]
        
        return "\n".join(sections)
    
    @staticmethod
    def get_qe_input_mc2d(formula: str, 
                        k_points_2d: Optional[List[int]],
                        pseudo_dir: Path) -> tuple[str, str, Atoms]:
        """
        Generate the Quantum Espresso input file for a specified chemical formula in 2D.

        Args:
            formula (str): The chemical formula for which to generate the input file.
            k_points_2d (Optional[List[int]]): A list of integers specifying the k-point grid for the 2D calculation.
                                                If None, a default grid of [7, 7, 2, 0, 0, 0] will be used.
            pseudo_dir (Path): The directory path where the pseudopotential files are located.

        Returns:
            tuple[str, str, Atoms]: A tuple containing:
                - The formatted Quantum Espresso input as a string.
                - A unique identifier (UUID) for the structure.
                - An Atoms object representing the atomic structure.
        """
        # set initial k_points
        if k_points_2d is None:
            k_points_2d = [7, 7, 2, 0, 0, 0]

        uuid = HelperRetrieverMc2d.get_uuid(formula)
        atoms = HelperRetrieverMc2d.uuid_atoms(uuid)
        qe_input = HelperRetrieverMc2d.format_qe_structure(atoms, k_points_2d, pseudo_dir)
        return qe_input, uuid, atoms
    

class HelperRetrieverMc3d:
    """
    A class to handle retrieval and formatting of 3D material structure data from MaterialsCloud.
    """
    @staticmethod
    def find_structure(formula: str, stoichiometric: bool = True) -> list:
        """
        Find the compound structure with the given chemical formula.
        
        Args:
            formula: Chemical formula to search for
            stoichiometric: Whether to enforce stoichiometric matching
            
        Returns:
            List containing compound data and formulas
        """
        find_out = find_compound(formula, stoichiometric=stoichiometric)
        if isinstance(find_out, str):
            raise ValueError(f"No stoichiometric compound found: {find_out}")
        return find_out

    @staticmethod
    def get_qe_input(formula: str) -> list[tuple[str, str]]:
        """
        Get Quantum Espresso input files for a given formula.
        
        Args:
            formula: Chemical formula to get input files for
            
        Returns:
            List of QE input file contents
        """
        inputs = qe_input_query(formula)
        if isinstance(inputs, str):
            raise ValueError(f"Failed to get the initial material data: {inputs}")
        return inputs

    @staticmethod
    def get_qe_input_mc3d(formula_fb: dict, pp_dir: Path) -> list[tuple[Atoms, int, str, str, str, str]]:
        """
        Retrieves and processes Quantum Espresso input data for a 3D material from Materials Cloud.

        This function takes a chemical formula and a path to the pseudopotential directory,
        retrieves the corresponding structure data from Materials Cloud, and then processes
        the Quantum Espresso input files associated with that structure.

        Args:
            formula_fb (dict): The chemical formula of the material with fallback, {'main': 'formula', 'fallback': 'formula'}
            pp_dir (Path): The path to the directory containing pseudopotential files.

        Returns:
            list[tuple[Atoms, int, str, str, str, str]]: A list of tuples, where each tuple contains:
                - atoms (ase.Atoms): An ASE Atoms object representing the structure.
                - symmetry.number (int): The space group number.
                - sg_dict.get(str(symmetry.number)) (str): The space group symbol.
                - qe_inp (str): The modified Quantum Espresso input string with corrected pseudopotential names.
                - carde_st (str): A string containing the processed card information.
                - uuid (str): A string representing the uuid of the structure.
        """
        if 'fallback' not in formula_fb:
            double_formula = parse_chemical_formula(formula_fb['main'])
            # double the counts of the elements
            double_formula = dict(map(lambda item: (item[0], item[1] * 2), double_formula.items()))

            # use double formula as fallback formula
            formula_fb['fallback'] = double_formula

        # Find formula data
        find_out = HelperRetrieverMc3d.find_structure(formula_fb['main'])[1]
        formula = formula_fb.get('main')
        if formula not in find_out:
            formula = formula_fb.get('fallback')
        if formula not in find_out:
            formula = find_out[0]

        print(formula)
        # Get QE inputs for the formula
        results = []
        qe_inputs = HelperRetrieverMc3d.get_qe_input(formula)
        for qe_input, uuid in qe_inputs:
            x_ = qe_polisher(formula, qe_input, pp_dir)
            results.append( [x_[0], x_[1], x_[2], x_[3], x_[4], uuid] )
        
        return results