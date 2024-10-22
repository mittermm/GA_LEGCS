import csv
import xml.etree.ElementTree as ET

# Convert nodes.csv to nod.xml
def csv_to_nod_xml(csv_file, xml_file):
    tree = ET.Element("nodes")
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ET.SubElement(tree, "node", id=row['id'], x=row['x'], y=row['y'])
    tree = ET.ElementTree(tree)
    tree.write(xml_file)

# Convert edges.csv to edg.xml
def csv_to_edg_xml(csv_file, xml_file):
    tree = ET.Element("edges")
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            edge_id = f"edge_{index}"
            ET.SubElement(tree,
                          "edge",
                          id=edge_id,
                          attrib={
                              "from": row['from'],
                              "to": row['to'],
                              "length": row['length'],
                              "speed": row['speed'],
                              "numLanes": row['numLanes']
                          })
    tree = ET.ElementTree(tree)
    tree.write(xml_file)

#csv_to_nod_xml('nodes.csv', 'nod.xml')
#csv_to_edg_xml('edges.csv', 'edg.xml')


