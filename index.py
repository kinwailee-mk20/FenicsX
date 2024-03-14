from generate_mesh import *
import matplotlib.pyplot as plt


mesh_size = 0.000625
r_obj = 0.005


points_on_proc, c_1 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=-0.06)
_ , c_2 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=-0.04)
_ , c_3 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=-0.02)
_ , c_4 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=0)
_ , c_5 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=0.02)
_ , c_6 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=0.04)
_ , c_7 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=0.06)
_ , c_8 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='conductor', K=0.08)

_ , i_1 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=-0.06)
_ , i_2 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=-0.04)
_ , i_3 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=-0.02)
_ , i_4 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=0)
_ , i_5 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=0.02)
_ , i_6 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=0.04)
_ , i_7 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=0.06)
_ , i_8 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='insulator', K=0.08)

_ , s_1 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=-0.06)
_ , s_2 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=-0.04)
_ , s_3 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=-0.02)
_ , s_4 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=0)
_ , s_5 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=0.02)
_ , s_6 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=0.04)
_ , s_7 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=0.06)
_ , s_8 = get_solution(r_obj=r_obj, mesh_size=mesh_size, conductivity='same', K=0.08)


diff_cond_1, diff_ins_1 = c_1 - s_1, i_1 - s_1
diff_cond_2, diff_ins_2 = c_2 - s_2, i_2 - s_2
diff_cond_3, diff_ins_3 = c_3 - s_3, i_3 - s_3
diff_cond_4, diff_ins_4 = c_4 - s_4, i_4 - s_4
diff_cond_5, diff_ins_5 = c_5 - s_5, i_5 - s_5
diff_cond_6, diff_ins_6 = c_6 - s_6, i_6 - s_6
diff_cond_7, diff_ins_7 = c_7 - s_7, i_7 - s_7
diff_cond_8, diff_ins_8 = c_8 - s_8, i_8 - s_8


# absolute potential - conductor
fig = plt.figure(constrained_layout=True)
plt.plot(points_on_proc[:, 0] * 1000, c_1 * 1000, "b", linewidth=2, label="-60")
plt.plot(points_on_proc[:, 0] * 1000, c_2 * 1000, "g", linewidth=2, label="-40")
plt.plot(points_on_proc[:, 0] * 1000, c_3 * 1000, "r", linewidth=2, label="-20")
plt.plot(points_on_proc[:, 0] * 1000, c_4 * 1000, "c", linewidth=2, label="0")
plt.plot(points_on_proc[:, 0] * 1000, c_5 * 1000, "m", linewidth=2, label="20")
plt.plot(points_on_proc[:, 0] * 1000, c_6 * 1000, "y", linewidth=2, label="40")
plt.plot(points_on_proc[:, 0] * 1000, c_7 * 1000, "k", linewidth=2, label="60")
plt.plot(points_on_proc[:, 0] * 1000, c_8 * 1000, "g", linewidth=2, label="80")

plt.grid(True)
plt.title("Conducting disk (radius = 5 mm)")
plt.ylabel("Absolute Potential (mV)")
plt.xlabel("z-coordinate (m)")
plt.legend()
plt.savefig("Absolute_conductor_disk_5.png")


fig = plt.figure(constrained_layout=True)
plt.plot(points_on_proc[:, 0] * 1000, i_1 * 1000, "b", linewidth=2, label="-60")
plt.plot(points_on_proc[:, 0] * 1000, i_2 * 1000, "g", linewidth=2, label="-40")
plt.plot(points_on_proc[:, 0] * 1000, i_3 * 1000, "r", linewidth=2, label="-20")
plt.plot(points_on_proc[:, 0] * 1000, i_4 * 1000, "c", linewidth=2, label="0")
plt.plot(points_on_proc[:, 0] * 1000, i_5 * 1000, "m", linewidth=2, label="20")
plt.plot(points_on_proc[:, 0] * 1000, i_6 * 1000, "y", linewidth=2, label="40")
plt.plot(points_on_proc[:, 0] * 1000, i_7 * 1000, "k", linewidth=2, label="60")
plt.plot(points_on_proc[:, 0] * 1000, i_8 * 1000, "g", linewidth=2, label="80")

plt.grid(True)
plt.title("Insulating disk (radius = 5 mm)")
plt.ylabel("Potential Difference (mV)")
plt.xlabel("z-coordinate (m)")
plt.legend()
plt.savefig("Absolute_insulator_disk_5.png")

# plotting
fig = plt.figure(constrained_layout=True)
plt.plot(points_on_proc[:, 0] * 1000, s_1 * 1000, "b", linewidth=2, label="-60")
plt.plot(points_on_proc[:, 0] * 1000, s_2 * 1000, "g", linewidth=2, label="-40")
plt.plot(points_on_proc[:, 0] * 1000, s_3 * 1000, "r", linewidth=2, label="-20")
plt.plot(points_on_proc[:, 0] * 1000, s_4 * 1000, "c", linewidth=2, label="0")
plt.plot(points_on_proc[:, 0] * 1000, s_5 * 1000, "m", linewidth=2, label="20")
plt.plot(points_on_proc[:, 0] * 1000, s_6 * 1000, "y", linewidth=2, label="40")
plt.plot(points_on_proc[:, 0] * 1000, s_7 * 1000, "k", linewidth=2, label="60")
plt.plot(points_on_proc[:, 0] * 1000, s_8 * 1000, "g", linewidth=2, label="80")

plt.grid(True)
plt.title("Invisible disk (radius = 5 mm)")
plt.ylabel("Potential Difference (mV)")
plt.xlabel("z-coordinate (m)")
plt.legend()
plt.savefig("Absolute_invisible_disk_5.png")


# plotting
fig = plt.figure(constrained_layout=True)
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_1 * 1000, "b", linewidth=2, label="-60")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_2 * 1000, "g", linewidth=2, label="-40")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_3 * 1000, "r", linewidth=2, label="-20")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_4 * 1000, "c", linewidth=2, label="0")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_5 * 1000, "m", linewidth=2, label="20")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_6 * 1000, "y", linewidth=2, label="40")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_7 * 1000, "k", linewidth=2, label="60")
plt.plot(points_on_proc[:, 0] * 1000, diff_cond_8 * 1000, "g", linewidth=2, label="80")

plt.grid(True)
plt.title("Conducting disk (radius = 5 mm)")
plt.ylabel("Potential Difference (mV)")
plt.xlabel("z-coordinate (m)")
plt.legend()
plt.savefig("Difference_conductor_disk_5.png")


fig = plt.figure(constrained_layout=True)
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_1 * 1000, "b", linewidth=2, label="-60")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_2 * 1000, "g", linewidth=2, label="-40")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_3 * 1000, "r", linewidth=2, label="-20")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_4 * 1000, "c", linewidth=2, label="0")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_5 * 1000, "m", linewidth=2, label="20")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_6 * 1000, "y", linewidth=2, label="40")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_7 * 1000, "k", linewidth=2, label="60")
plt.plot(points_on_proc[:, 0] * 1000, diff_ins_8 * 1000, "g", linewidth=2, label="80")

plt.grid(True)
plt.title("Insulating disk (radius = 5 mm)")
plt.ylabel("Potential Difference (mV)")
plt.xlabel("z-coordinate (m)")
plt.legend()
plt.savefig("Difference_insulator_disk_5.png")


# save data
conductor = np.column_stack((c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8))
insulator = np.column_stack((i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8))
same = np.column_stack((s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8))
diff_cond = np.column_stack((diff_cond_1, diff_cond_2, diff_cond_3, diff_cond_4, diff_cond_5, diff_cond_6, diff_cond_7, diff_cond_8))
diff_ins = np.column_stack((diff_ins_1, diff_ins_2, diff_ins_3, diff_ins_4, diff_ins_5, diff_ins_6, diff_ins_7, diff_ins_8))
np.savetxt("cond_disk_5.csv", conductor, delimiter=",")
np.savetxt("ins_disk_5.csv", insulator, delimiter=",")
np.savetxt("same_disk_5.csv", same, delimiter=",")
np.savetxt("diff_cond_disk_5.csv", diff_cond, delimiter=",")
np.savetxt("diff_ins_disk_5.csv", diff_ins, delimiter=",")
