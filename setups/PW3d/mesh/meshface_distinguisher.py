import xml.etree.ElementTree as ET

name = "cyl10x2cm.xml"
#name = "cyl25x4mm.xml"

tree = ET.parse(name)
root = tree.getroot()

mesh = root[0]
vertices = mesh[0]
cells = mesh[1]
domains = mesh[2]
mesh_value_collection_2 = domains[0]



for value in mesh_value_collection_2:
    if value.get('value') == '10':
        cell_index = int(value.get('cell_index'))
        v0 = int(cells[cell_index].get('v0'))
        if float(vertices[v0].get('y')) > 0:
            value.set('value', '11')

tree.write(name)
