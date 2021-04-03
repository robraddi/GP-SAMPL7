import GPR
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from IPython.display import Image
pd.options.display.max_columns = 50
pd.options.display.max_rows = None

SAMPL="SAMPL6"
model="stdGP"#"deepGP"#"stdGP"#
input_files = GPR.toolbox.get_files(f"../Structures/{SAMPL}/input_files/SM*/SM*.csv")
feature_files = GPR.toolbox.get_files(f"../predictions/{SAMPL}_{model}/SM*/total_feature_input.pkl")
results_files = GPR.toolbox.get_files(f"../predictions/{SAMPL}_{model}/SM*/*_results.pkl")
transitions_files = GPR.toolbox.get_files(f"../Structures/{SAMPL}/input_files/SM*/SM*_transitions.npy")
micro_pkas,macro_pKas = [],[]
for i,input_file in enumerate(input_files):
    results = pd.read_pickle(results_files[i])
    micro_pkas.append(results)
    SM = results_files[i].split("/")[-2]
    in_file = pd.read_csv(input_file, index_col=False)
    max_charge = np.max(list(in_file["charge state"]))
    min_charge = np.min(list(in_file["charge state"]))
    charge = max_charge
    pKas = {}
    pKas["Molecule"] = SM
    while charge >= min_charge+1:
        #print(input_file, results_files[i], feature_files[i])
        try:
            pKas[charge] = GPR.Processing.get_macroscopic_pKas(int(charge), input_file, results_files[i], feature_files[i])
        except(Exception) as e: pass
        charge -= 1
    macro_pKas.append(pKas)

macro_pKas = pd.DataFrame(macro_pKas)
macro_pKas = macro_pKas[[0,1,2]]
#macro_pKas.to_csv(f"{model}/macropKa_predictions.csv", index=False)
macro_pKas

micro_pkas = pd.concat(micro_pkas)
micro_pkas = micro_pkas.drop(['small molecule'], axis=1)
micro_pkas = micro_pkas.round(2)
micro_pkas['protonated'] = [x.replace("_micro"," m") for x in micro_pkas['protonated'].to_numpy()]
micro_pkas['deprotonated'] = [x.replace("_micro"," m") for x in micro_pkas['deprotonated'].to_numpy()]
micro_pkas['posterior'] = micro_pkas['posterior'].apply(str)
micro_pkas['posterior_std'] = micro_pkas['posterior_std'].apply(str)
micro_pkas['prediction'] = "$"+micro_pkas['posterior']+"\pm"+micro_pkas['posterior_std']+"$"
micro_pkas = micro_pkas.drop(['posterior', 'posterior_std'], axis=1)
micro_pkas


Gs = []
for i,input_file in enumerate(input_files):
    results = pd.read_pickle(results_files[i])
    if any(np.isnan(np.array(list(results["posterior"])))): print(i);continue
    SM = results_files[i].split("/")[-2]
    in_file = pd.read_csv(input_file, index_col=False)
    ref = list(in_file["microstate ID"])[0]
    free_energies = GPR.Processing.get_Gs(ref, input_file, results_files[i], feature_files[i])
    free_energies = GPR.Processing.fix_Gs(free_energies, input_file, results_files[i], feature_files[i])
    Gs.append(free_energies)
Gs = pd.concat(Gs)
col_order = ['Ref', 'Microstate ID', 'Charge', '∆G', '∆G std']
Gs = Gs[col_order]
Gs.to_csv(f"{model}/free_energy_predictions.csv", index=False)

network_imgs = GPR.toolbox.get_files(f"{model}/SM*/SM*_network.png")
for i in range(len(input_files)):
    print(f'Small Molecule: {network_imgs[i].split("/Network_of_states.png")[0].split("/")[-1]}')
    display(Image(network_imgs[i]))
    df = pd.read_csv(input_files[i])
    display(HTML(df.to_html()))
    df2 = pd.read_pickle(feature_files[i])
    df1 = pd.read_pickle(results_files[i])
    df = pd.concat([df1,df2], axis=1)
    display(HTML(df.to_html()))
    print("\n\n")
