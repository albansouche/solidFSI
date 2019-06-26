# !/usr/bin/python

from __future__ import print_function  # Python 2 / 3 support

# Local imports
from vmtkmeshgeneratorfsi import *

from os import path
from IPython import embed
from vmtk import vtkvmtk, vmtkscripts
import vtk
import json
import argparse

from IPython import embed


# Inputs #######################################################################
ifile_surface = "meshes/cyl10x2cm.vtp"
ofile_mesh = "meshes/cyl10x2cm"
TargetEdgeLength = 0.002
Thick_solid = 0.002
BoundaryLayerThicknessFactor = Thick_solid / TargetEdgeLength  # Wall Thickness == TargetEdgeLength*BoundaryLayerThicknessFactor
################################################################################


# Read vtp surface file (vtp) ##################################################
reader = vmtkscripts.vmtkSurfaceReader()
reader.InputFileName = ifile_surface
reader.Execute()
surface = reader.Surface
################################################################################


# Add flow extensions ##########################################################
flow_Extension_action = 0
if flow_Extension_action:
    print("--- Adding flow extensions")
    extender = vmtkscripts.vmtkFlowExtensions()
    extender.Surface = surface
    extender.AdaptiveExtensionLength = 1
    extender.ExtensionRatio = 3
    extender.ExtensionMode = "boundarynormal"
    extender.CenterlineNormalEstimationDistanceRatio = 1.0
    extender.Interactive = 0
    extender.Execute()
    surface = extender.Surface
################################################################################


# Create FSI mesh ##############################################################
print("--- Creating fsi mesh")
meshGenerator = vmtkMeshGeneratorFsi()
meshGenerator.Surface = surface
# for remeshing
meshGenerator.SkipRemeshing = 0
meshGenerator.ElementSizeMode = 'edgelength'
meshGenerator.TargetEdgeLength = TargetEdgeLength
meshGenerator.MaxEdgeLength = 1.5*meshGenerator.TargetEdgeLength
meshGenerator.MinEdgeLength = 0.5*meshGenerator.TargetEdgeLength
# for boundary layer (used for both fluid boundary layer and solid domain)
meshGenerator.BoundaryLayer = 1
meshGenerator.NumberOfSubLayers = 2
meshGenerator.BoundaryLayerOnCaps = 0
meshGenerator.SubLayerRatio = 1
meshGenerator.BoundaryLayerThicknessFactor = BoundaryLayerThicknessFactor
# mesh
meshGenerator.Tetrahedralize = 1
# Cells and walls numbering
meshGenerator.SolidSideWallId = 11
meshGenerator.InterfaceId_fsi = 22
meshGenerator.InterfaceId_outer = 33
meshGenerator.VolumeId_fluid = 0  # (keep to 0)
meshGenerator.VolumeId_solid = 1
meshGenerator.Execute()
mesh = meshGenerator.Mesh
################################################################################


# Write mesh in VTU format #####################################################
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(path.join(ofile_mesh + ".vtu"))
writer.SetInputData(mesh)
writer.Update()
writer.Write()
################################################################################


# Write mesh to FEniCS to format ###############################################
meshWriter = vmtkscripts.vmtkMeshWriter()
meshWriter.CellEntityIdsArrayName = "CellEntityIds"
meshWriter.Mesh = mesh
meshWriter.OutputFileName = path.join(ofile_mesh + ".xml")
meshWriter.WriteRegionMarkers = 1
meshWriter.Execute()
################################################################################


# remeshing = vmtkscripts.vmtkSurfaceRemeshing()
# remeshing.Surface = surface
# remeshing.ElementSizeMode  = 'edgelength'
# remeshing.TargetEdgeLength = 0.045
# remeshing.MaxEdgeLength    = 1.1*remeshing.TargetEdgeLength
# remeshing.MinEdgeLength    = 0.9*remeshing.TargetEdgeLength
# remeshing.Execute()
# writer = vtk.vtkXMLPolyDataWriter()
# writer.SetFileName("meshes/toto.vtp")
# writer.SetInputData(remeshing.Surface)
# writer.Update()
# writer.Write()
