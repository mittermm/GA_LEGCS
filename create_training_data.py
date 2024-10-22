from gen_networks.utils import idf_to_str
from config.run_simulation import run_simulation
import shutil

for i in range(10):
    idf = idf_to_str(i)

    shutil.copyfile("gen_networks/networks/" + idf + ".net.xml", "config/current_network.net.xml")
    avg_distance, avg_speed = run_simulation("config/base.sumo.cfg")

    print("network " + idf + ": "
          + "\n\taverage distance: " + str(avg_distance)
          + "\n\taverage speed:    " + str(avg_speed))

    with open("training_data/data.csv", 'a') as file:
        file.write(idf + "," + str(avg_distance) + "," + str(avg_speed)+ "\n")
