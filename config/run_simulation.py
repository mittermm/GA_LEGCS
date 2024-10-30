import traci
import sumolib

def run_simulation(sumo_config_path):
    # Start the SUMO simulation
    sumo_cmd = ["sumo", "-c", sumo_config_path, "--junction-taz", "--no-warnings"]
    traci.start(sumo_cmd)
    
    total_avg_speed = 0
    steps = 0
    
    vehicle_distances = {}    
   
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()  # Advance the simulation by one step
            
            # Get the list of all vehicles in the simulation at the current step
            vehicle_ids = traci.vehicle.getIDList()
            
            # Calculate the speed for each vehicle
            total_speed = 0
            vehicle_count = 0
            for vehicle_id in vehicle_ids:
                total_speed += traci.vehicle.getSpeed(vehicle_id)
                vehicle_distances[vehicle_id] = traci.vehicle.getDistance(vehicle_id)
                vehicle_count += 1
            if vehicle_count > 0:
                total_avg_speed += total_speed / vehicle_count
                steps += 1
                
            
                
    finally:
        # Close the SUMO simulation
        traci.close()
    
    average_speed = total_avg_speed / steps
    average_distance = sum(vehicle_distances.values()) / len(vehicle_distances)

    return average_distance, average_speed

