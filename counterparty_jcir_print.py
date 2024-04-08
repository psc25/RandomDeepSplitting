import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})

path = os.path.join(os.getcwd(), "counterparty_jcir/")
dd = [10, 50, 100, 500, 1000, 5000, 10000]
runs = 10

mlp_sol = np.zeros([len(dd), runs])
mlp_tms = np.zeros([len(dd), runs])
mlp_fev = np.zeros([len(dd), runs])
det_sol = np.zeros([len(dd), runs])
det_tms = np.zeros([len(dd), runs])
det_fev = np.zeros([len(dd), runs])
rnd_sol = np.zeros([len(dd), runs])
rnd_tms = np.zeros([len(dd), runs])
rnd_fev = np.zeros([len(dd), runs])
for i in range(len(dd)):
    mlp_sol[i] = np.loadtxt(path + "mlp_sol_" + str(dd[i]) + ".csv")[-1]
    mlp_tms[i] = np.loadtxt(path + "mlp_tms_" + str(dd[i]) + ".csv")[-1]
    mlp_fev[i] = np.loadtxt(path + "mlp_fev_" + str(dd[i]) + ".csv")[-1]
    det_sol[i] = np.loadtxt(path + "det_sol_" + str(dd[i]) + ".csv")
    det_tms[i] = np.loadtxt(path + "det_tms_" + str(dd[i]) + ".csv")
    det_fev[i] = np.loadtxt(path + "det_fev_" + str(dd[i]) + ".csv")
    rnd_sol[i] = np.loadtxt(path + "rnd_sol_" + str(dd[i]) + ".csv")
    rnd_tms[i] = np.loadtxt(path + "rnd_tms_" + str(dd[i]) + ".csv")
    rnd_fev[i] = np.loadtxt(path + "rnd_fev_" + str(dd[i]) + ".csv")

err_detm = np.nanmean(np.abs(det_sol - np.nanmean(mlp_sol, axis = 1, keepdims = True)), axis = 1)
err_rand = np.nanmean(np.abs(rnd_sol - np.nanmean(mlp_sol, axis = 1, keepdims = True)), axis = 1)

# Table
txt = ""
for di in range(len(dd)):
    txt = txt + str(dd[di]) + " & " + '{:.4f}'.format(np.nanmean(det_sol[di])) + " & \cellcolor{lightgray} " + '{:.4f}'.format(np.nanmean(rnd_sol[di])) + " & " + '{:.4f}'.format(np.nanmean(mlp_sol[di])) 
    f1 = np.nanmean(det_fev[di])
    if f1 > 0:
        e1 = np.floor(np.log10(f1))
        r1 = f1/np.power(10, e1)
    else:
        e1 = 0
        r1 = 0
        
    f2 = np.nanmean(rnd_fev[di])
    if f2 > 0:
        e2 = np.floor(np.log10(f2))
        r2 = f2/np.power(10, e2)
    else:
        e2 = 0
        r2 = 0
        
    f3 = np.nanmean(mlp_fev[di])
    if f3 > 0:
        e3 = np.floor(np.log10(f3))
        r3 = f3/np.power(10, e3)
    else:
        e3 = 0
        r3 = 0
        
    txt = txt + " & " + '{:.2f}'.format(np.nanmean(det_tms[di])) + " & \cellcolor{lightgray} " + '{:.2f}'.format(np.nanmean(rnd_tms[di])) + " & " + '{:.2f}'.format(np.nanmean(mlp_tms[di])) + " \\\ \n"
    txt = txt + " & \\textit{" + '{:.4f}'.format(np.nanstd(det_sol[di])) + "} & \cellcolor{lightgray} \\textit{" + '{:.4f}'.format(np.nanstd(rnd_sol[di])) + "} & \\textit{" + '{:.4f}'.format(np.nanstd(mlp_sol[di])) + "}"
    txt = txt + " & $" + '{:.2f}'.format(r1) + " \\cdot 10^{" + '{:.0f}'.format(e1) + "}$ & \cellcolor{lightgray} $" + '{:.2f}'.format(r2) + " \\cdot 10^{" + '{:.0f}'.format(e2) + "}$ & $" + '{:.2f}'.format(r3) + " \\cdot 10^{" + '{:.0f}'.format(e3) + "}$ \\\ \n"
    txt = txt + "\hline \n"
    
text_file = open(path + "table.txt", "w")
n = text_file.write(txt[:-13])
text_file.close()