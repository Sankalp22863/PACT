import numpy as np

# Calulate the thermal resistance and capacitance for each grid cell
# Single mode


def defineGridProperties(length, height, thickness, properties):
    ro = float(properties['thermalresistivity ((m-k)/w)'])
    sp = float(properties['specificheatcapacity (j/m^3k)'])
    # Optional anisotropy: 'inplanethermalresistivity ((m-k)/w)' sets the
    # lateral (x/y) resistivity; the standard entry then applies to the
    # vertical (z) path only. Falls back to isotropic when absent.
    ro_ip = float(properties.get('inplanethermalresistivity ((m-k)/w)', ro))

    Rx = ro_ip*length/(height*thickness)
    Ry = ro_ip*height/(length*thickness)
    Rz = ro*thickness/(length*height)
    Conv = 0
    # use this capacitance to validate with COMSOL
    Capacitance=1*sp*length*height*thickness
    # use this capacitance to validate with Hotspot (Based on the https://github.com/uvahotspot/HotSpot/blob/master/RCutil.c#L29)
    # Capacitance = 0.33*sp*thickness*length*height

    I = 0
    direction = 'z'
    out = {"Rx": Rx, "Ry": Ry, "Rz": Rz, "Capacitance": Capacitance,
           "I": I, "direction": direction, "Conv": Conv}
    return out
# Matrix mode


def defineGridPropertiesMatrix(length, height, thickness, properties):
    grid_rows = properties['grid_rows']
    grid_cols = properties['grid_cols']
    ro = float(properties['thermalresistivity ((m-k)/w)'])
    sp = float(properties['specificheatcapacity (j/m^3k)'])
    ro_ip = float(properties.get('inplanethermalresistivity ((m-k)/w)', ro))
    rx = ro_ip*length/(height*thickness)
    ry = ro_ip*height/(length*thickness)
    rz = ro*thickness/(length*height)
    # cap=1*sp*length*height*thickness
    Rx = np.full((grid_rows, grid_cols), rx)
    Ry = np.full((grid_rows, grid_cols), ry)
    Rz = np.full((grid_rows, grid_cols), rz)
    #Capacitance = np.full((grid_rows, grid_cols),cap)
    #Capacitance = np.zeros((grid_rows,grid_cols))

    # use this capacitance to validate with COMSOL
    capacitance = 1*sp*length*height*thickness
    # use this capacitance to validate with Hotspot (Based on the https://github.com/uvahotspot/HotSpot/blob/master/RCutil.c#L29)
    # capacitance = 0.33*sp*thickness*length*height
    Capacitance = np.full((grid_rows, grid_cols), capacitance)
    Conv = 0
    print('Zihao matrix cap ********************', capacitance)

    I = np.zeros((grid_rows, grid_cols))
    direction = '0'
    out = {"Rx": Rx, "Ry": Ry, "Rz": Rz, "Capacitance": Capacitance,
           "I": I, "direction": direction, "Conv": Conv}
    return out
