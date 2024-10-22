import random

towns = [
            "dublin","dublin","dublin","dublin","dublin","dublin",
            "dublin","dublin","dublin","dublin","dublin","dublin",
            "belfast","belfast","belfast","cork","cork","cork",
            "belfast","belfast","belfast","cork","cork","cork",
            "limerick","limerick","derry","derry","galway","galway",
            "limerick","limerick","derry","derry","galway","galway",
            
            "antrim","craigavon","carlow","cavan","ennis","letterkenny",
            "enniskillen","tralee","naas","kilkenny","portlaoise","carrick",
            "longford","dundalk","castlebar","navan","monaghan","tullamore",
            "roscommon","sligo","clonmel","omagh","waterford","athlone",
            "wexford","bray"
        ]

coast_towns = [
            "derry", "antrim", "belfast", "dundalk", "dublin", "bray",
            "wexford", "waterford", "cork", "tralee", "limerick", "ennis",
            "galway", "castlebar", "sligo", "letterkenny"
        ]

print('<?xml version="1.0" encoding="UTF-8"?>\n\n<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')

def random_routes(routes_number):
    for c in range(routes_number):
        town1 = towns[random.randint(0, len(towns) -1)]
        town2 = town1
        while town1 == town2:
            town2 = towns[random.randint(0, len(towns) -1)]
    
        d = (1.0 * c) / 10
        print('\t<trip id="t_' + str(c) + '" depart="' + str(d) + '" fromJunction="' + town1 + '" toJunction="' + town2 + '" />')

#scenario 1: all ireland hurling final - traffic from ennis and cork to dublin
def scenario1(routes_number):
    for c in range(routes_number):
        if c % 2 == 0:
            town1 = "ennis"
        else:
            town1 = "cork"
        town2 = "dublin"

        d = (1.0 * c) / 10
        print('\t<trip id="tsc1_' + str(c) + '" depart="' + str(d) + '" fromJunction="' + town1 + '" toJunction="' + town2 + '" />')

#scenario 2: ed sheeran concert in thomond park - traffic from all over ireland to limerick
def scenario2(routes_number):
    for c in range(routes_number):
        town1 = "limerick"
        town2 = town1
        while town1 == town2:
            town1 = towns[random.randint(0, len(towns) -1)]
    
        d = (1.0 * c) / 10
        print('\t<trip id="tsc2_' + str(c) + '" depart="' + str(d) + '" fromJunction="' + town1 + '" toJunction="' + town2 + '" />')    

#scenario 3: beach day - traffic to towns at the coast
def scenario3(routes_number):        
    for c in range(routes_number):
        town1 = towns[random.randint(0, len(towns) -1)]
        town2 = town1
        while town1 == town2 or town2 not in coast_towns:
            town2 = towns[random.randint(0, len(towns) -1)]
    
        d = (1.0 * c) / 10
        print('\t<trip id="tsc3_' + str(c) + '" depart="' + str(d) + '" fromJunction="' + town1 + '" toJunction="' + town2 + '" />')


random_routes(1000)
#scenario1(200)
#scenario2(200)
#scenario3(500)

print("</routes>")

