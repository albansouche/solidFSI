import xml.etree.ElementTree as ET

# Input mesh location
name = "cyl10x2cm.xml"
#name = "cyl25x4mm.xml"

output_name = name  # Same output name as input => overwrite mesh
#output_name = "distinguished_mesh.xml"

tree = ET.parse(name)
root = tree.getroot()

mesh = root[0]
vertices = mesh[0]
cells = mesh[1]
domains = mesh[2]
mesh_value_collection_2 = domains[0]


# We consider a cylinder centered in (0,0,0), with y-axis as axis.
# Boundaries y=-0.05 and y=+0.05 are marked with '10',
# we wish to distinguish them, i.e. marking boundary y=+0.05 with '11'
# We may separate the two solely based on the sign of y
for value in mesh_value_collection_2:
    if value.get('value') == '10':
        cell_index = int(value.get('cell_index'))
        v0 = int(cells[cell_index].get('v0'))
        if float(vertices[v0].get('y')) > 0:  # 0.05 > 0, -0.05 < 0
            value.set('value', '11')

# Save modified mesh
tree.write(name)
