import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})

path = os.path.join(os.getcwd(), "default_merton/")
dd = [10, 50, 100, 500, 1000, 5000, 10000]
runs = 10

col1 = ["#006795", "#AF3235", "#221E1F"]
col2 = ["MidnightBlue", "Maroon", "Black"]

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

# Table
txt = ""
for di in range(len(dd)):
    txt = txt + str(dd[di]) + " & \\textcolor{" + col2[0] + "}{" + '{:.4f}'.format(np.nanmean(det_sol[di])) + "} & \\textcolor{" + col2[1] + "}{" + '{:.4f}'.format(np.nanmean(rnd_sol[di])) + "} & \\textcolor{" + col2[2] + "}{" + '{:.4f}'.format(np.nanmean(mlp_sol[di])) + "}"
    txt = txt + " & \\textcolor{" + col2[0] + "}{" + '{:.2f}'.format(np.nanmean(det_tms[di])) + "} & \\textcolor{" + col2[1] + "}{" + '{:.2f}'.format(np.nanmean(rnd_tms[di])) + "} & \\textcolor{" + col2[2] + "}{" + '{:.2f}'.format(np.nanmean(mlp_tms[di])) + "} \\\ \n"
    txt = txt + " & \\textcolor{" + col2[0] + "}{" + '{:.4f}'.format(np.nanstd(det_sol[di])) + "} & \\textcolor{" + col2[1] + "}{" + '{:.4f}'.format(np.nanstd(rnd_sol[di])) + "} & \\textcolor{" + col2[2] + "}{" + '{:.4f}'.format(np.nanstd(mlp_sol[di])) + "}"
    
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
        
    txt = txt + " & \\textcolor{" + col2[0] + "}{$" + '{:.2f}'.format(r1) + " \\cdot 10^{" + '{:.0f}'.format(e1) + "}$} & \\textcolor{" + col2[1] + "}{$" + '{:.2f}'.format(r2) + " \\cdot 10^{" + '{:.0f}'.format(e2) + "}$} & \\textcolor{" + col2[2] + "}{$" + '{:.2f}'.format(r3) + " \\cdot 10^{" + '{:.0f}'.format(e3) + "}$} \\\ \n"
    txt = txt + "\hline \n"
    
text_file = open(path + "table.txt", "w")
n = text_file.write(txt[:-13])
text_file.close()

# Figure 1: Solution and Standard Deviation
plt.rcParams.update({'legend.fontsize': 11,
                     'axes.labelsize': 12,
                     'axes.titlesize': 14,
                     'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

fig = plt.figure().set_figheight(5)
plt.errorbar(np.exp(-0.2)*np.array(dd), np.mean(mlp_sol, -1), 10.0*np.std(mlp_sol, -1), color = col1[2], linestyle = 'None', linewidth = 1.3, marker = 'o', markersize = 6)
plt.errorbar(1.0*np.array(dd), np.mean(det_sol, -1), 10.0*np.std(det_sol, -1), color = col1[0], linestyle = 'None', linewidth = 1.3, marker = 'o', markersize = 6)
plt.errorbar(np.exp(0.2)*np.array(dd), np.mean(rnd_sol, -1), 10.0*np.std(rnd_sol, -1), color = col1[1], linestyle = 'None', linewidth = 1.3, marker = 'o', markersize = 6)
plt.xscale('log')
plt.xticks(ticks = dd, labels = dd)

plt.plot(np.nan, np.nan, color = col1[0], linestyle = "-", linewidth = 1.3, marker = 'o', markersize = 6, label = "Determ.")
plt.plot(np.nan, np.nan, color = col1[1], linestyle = "-", linewidth = 1.3, marker = 'o', markersize = 6, label = "Random")
plt.plot(np.nan, np.nan, color = col1[2], linestyle = "-", linewidth = 1.3, marker = 'o', markersize = 6, label = "MLP")
plt.plot(np.nan, np.nan, color = "black", linestyle = None, marker = 'o', markersize = 6, label = "Solution")
plt.plot(np.nan, np.nan, color = "black", linestyle = "-", marker = None, label = "$\pm 10 \cdot $ Std. Dev.")
plt.legend(loc = "upper right", ncol = 2)
plt.xlabel("Dimension $d$")
plt.ylabel("Solution $u(t,x)$")
plt.title("Average Solution")

plt.savefig(path + "sol_std.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Figure 2: Running Time and Function Evaluation
fig, ax1 = plt.subplots()
fig.set_figheight(5)
ax1.plot(dd, np.mean(mlp_tms, -1), color = col1[2], linestyle = "-", marker = "<")
ax1.plot(dd, np.mean(det_tms, -1), color = col1[0], linestyle = "-", marker = "<")
ax1.plot(dd, np.mean(rnd_tms, -1), color = col1[1], linestyle = "-", marker = "<")
ax1.set_xlabel('Dimension $d$')
ax1.set_ylabel('Time (in seconds)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(dd)
ax1.set_xticklabels(dd)
ax1.set_ylim([0.8*np.min([np.mean(mlp_tms, -1), np.mean(det_tms, -1), np.mean(rnd_tms, -1)]), 
              3.0*np.max([np.mean(mlp_tms, -1), np.mean(det_tms, -1), np.mean(rnd_tms, -1)])])

ax2 = ax1.twinx()
ax2.plot(dd, np.mean(mlp_fev, -1), color = col1[2], linestyle = ":", marker = ">")
ax2.plot(dd, np.mean(det_fev, -1), color = col1[0], linestyle = ":", marker = ">")
ax2.plot(dd, np.mean(rnd_fev, -1), color = col1[1], linestyle = ":", marker = ">")
ax2.set_ylabel('Function Evaluations')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks(dd)
ax2.set_xticklabels(dd)
ax2.set_ylim([0.5*np.min([np.mean(mlp_fev, -1), np.mean(det_fev, -1), np.mean(rnd_fev, -1)]), 
              5.0*np.max([np.mean(mlp_fev, -1), np.mean(det_fev, -1), np.mean(rnd_fev, -1)])])

ax1.plot(np.nan, np.nan, color = col1[0], linestyle = "-", label = "Determ.")
ax1.plot(np.nan, np.nan, color = col1[1], linestyle = "-", label = "Random")
ax1.plot(np.nan, np.nan, color = col1[2], linestyle = "-", label = "MLP")
ax1.plot(np.nan, np.nan, color = "black", linestyle = "-", marker = "<", label = "Time")
ax1.plot(np.nan, np.nan, color = "black", linestyle = ":", marker = ">", label = "Fct. Eval.")
ax1.legend(loc = "upper left", ncol = 2)
ax1.set_title("Average Time & Evaluations")
plt.savefig(path + "tms_fev.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)