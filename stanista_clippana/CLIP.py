
import arcpy
from arcpy import env
env.workspace="D:\Stef\ARCPY"
ulazni_shp="D:\Stef\ARCPY\POVS\KLIP_KS_SVI_POVS\CLIP_HR2001012_Licko polje.shp"
#upisati path shpa s kojim klipas
klipam_sa_shp="D:\Stef\ARCPY\Licko_senjska\prenamjene\Merge_Polygon.shp"
izlazni="D:\Stef\ARCPY\Licko_senjska\prenamjene\klip_ks_LKP_prenamjene.shp"
xy_tolerance = ""
arcpy.Clip_analysis(ulazni_shp, klipam_sa_shp, izlazni, xy_tolerance)
