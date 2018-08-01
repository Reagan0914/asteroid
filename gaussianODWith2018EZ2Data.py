"""

GAUSSIAN ORBIT DETERMINATION of 4055 Magellan

"""

import matplotlib.pyplot as plt
import numpy as np

from numpy import *

from poliastro.twobody import Orbit, classical
from poliastro.bodies import Earth, Sun
from poliastro.neos import neows
from poliastro.plotting import OrbitPlotter
from astropy import units, time

# download higher precision ephemerides
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set("jpl")

from jplephem.spk import SPK
kernel = SPK.open('C:\\Users\\user\\Downloads\\de430.bsp')

# Constants
k = 0.017202099  # AU^(3/2)/solar day
c = 173.1446    # AU/day
r = 6371        # This is in km - convert to AU later
km2au = 1/149597870700e-3 #  km/au

def toDecimal(sexagismal):
    # Convert RA and Dec to decimal hours and degrees if in sexagismal
    if sexagismal[0] > 1:
        time = sexagismal[0]
        time += (sexagismal[1] / float(60))
        time += (sexagismal[2] / float(3600))
    else:
        time = sexagismal[0]
        time -= (sexagismal[1] / float(60))
        time -= (sexagismal[2] / float(3600))
    return time


def pHat(ra, dec):
    # Finds the rho hat vectors from the RA and dec
    pHatx = cos(ra) * cos(dec)
    pHaty = sin(ra) * cos(dec)
    pHatz = sin(dec)
    pHatVector = [pHatx, pHaty, pHatz]
    return pHatVector


def dotProduct(a, b):
    # Dot product
    total = 0
    for element in range(len(a)):
        total += a[element] * b[element]
    return total


# Cross product
def vcrossProduct(vec1, vec2):
    a = vec1[0]
    b = vec1[1]
    c = vec1[2]
    d = vec2[0]
    e = vec2[1]
    f = vec2[2]
    cross = [b * f - c * e, -(a * f - c * d), a * e - b * d]
    return cross


def tripleProduct(cross1, cross2, dot1):
    # Finds the triple product of (cross1 x cross2) * dot1
    cross = vcrossProduct(cross1, cross2)
    triple = dotProduct(cross, dot1)
    return triple


def rCalculate(p, pHat, R):
    # Calculates initial guess for r1, r2, r3
    r = [p * pHat[0] - R[0], p * pHat[1] - R[1], p * pHat[2] - R[2]]
    return r


def equatorialtoEcliptic(equatorial):
    # Rotates equatorial coordinates to ecliptic coordinates
    epsilon = 23.45027755 * pi / 180
    xeq = equatorial[0]
    yeq = equatorial[1]
    zeq = equatorial[2]
    xec = xeq
    yec = yeq * cos(epsilon) + zeq * sin(epsilon)
    zec = yeq * -sin(epsilon) + zeq * cos(epsilon)
    ecliptic = [xec, yec, zec]
    return ecliptic


"""
INPUTS
"""
# 2018 EZ2 Actual Data
ra1 = (8 + 44/60 + 55.96/3600) * pi / 180  # Converted to radians
dec1 = (-7 + 24/60 + 39.8/3600) * pi / 180  # CTR
t1 = 2.458189920830000e+06 # 2018 03 12.42083  # UTC
ra2 = (8 + 44/60 + 56.33) * pi / 180   # CTR
dec2 = (-7 + 25/60 + 3.7/3600) * pi / 180  # CTR
t2 = 2.458189922220000e+06 # 2018 03 12.42222  # UTC
ra3 = (8 + 46/60 + 0.86/3600) * pi / 180   # CTR
dec3 = (-8 + 30/60 + 51.6/3600) * pi / 180  # CTR
t3 = 2.458190129060000e+06 # 2018 03 12.62906  # UTC

# Get Earth positions in J2000 (DE430)
R1 = kernel[0,3].compute(t1) * km2au
R2 = kernel[0,3].compute(t2) * km2au
R3 = kernel[0,3].compute(t3) * km2au

"""
INITIAL GUESSES
"""

# Solve for pHat vectors
pHat1 = pHat(ra1, dec1)
pHat2 = pHat(ra2, dec2)
pHat3 = pHat(ra3, dec3)

print ("Rho hats: ")
print (pHat1)
print (pHat2)
print (pHat3)
print


# Time to tau conversion
tau1 = k * (t1 - t2)
tau2 = k * (t3 - t1)
tau3 = k * (t3 - t2)

print ("Taus: ", tau1, tau2, tau3)
print


# Initial guesses for a1 and a3
a1 = abs(tau3 / tau2)
a3 = abs(tau1 / tau2)

print ("a1: ", a1, " a3: ", a3)
print


# Find p1, p2, and p3
p1 = (a1 * tripleProduct(R1, pHat2, pHat3) - tripleProduct(R2, pHat2, pHat3)
      + a3 * tripleProduct(R3, pHat2, pHat3)) / (a1 * tripleProduct(pHat1, pHat2, pHat3))
p2 = (a1 * tripleProduct(pHat1, R1, pHat3) - tripleProduct(pHat1, R2, pHat3)
      + a3 * tripleProduct(pHat1, R3, pHat3)) / (-1. * tripleProduct(pHat1, pHat2, pHat3))
p3 = (a1 * tripleProduct(pHat2, R1, pHat1) - tripleProduct(pHat2, R2, pHat1)
      + a3 * tripleProduct(pHat2, R3, pHat1)) / (a3 * tripleProduct(pHat2, pHat3, pHat1))

print ("Rho scalars: ", p1, p2, p3)
print


# Calculate r vectors
r1 = rCalculate(p1, pHat1, R1)
r2 = rCalculate(p2, pHat2, R2)
r3 = rCalculate(p3, pHat3, R3)
rList = [r1, r3]
r2magnitude = linalg.norm(r2)

print ("r1: ", r1)
print ("r2: ", r2)
print ("r3: ", r3)
print ("r2 magnitude: ", r2magnitude)
print


# Calculate r0Dot
r2Dot = [(r3[0] - r1[0]) / tau2, (r3[1] - r1[1]) / tau2, (r3[2] - r1[2]) / tau2]
r2Dotmagnitude = linalg.norm(r2Dot)

print ("r2 dot: ", r2Dot)
print ("r2 dot magnitude: ", r2Dotmagnitude)
print


# Calculate f and g
f1 = 1. - (tau1)**2 / (2 * r2magnitude**3)
f3 = 1. - (tau3)**2 / (2 * r2magnitude**3)
g1 = tau1 - tau1**3 / (6. * r2magnitude**3)
g3 = tau3 - tau3**3 / (6. * r2magnitude**3)
fandGlist = [f1, g1, f3, g3]

print ("f1: ", f1, " f3: ", f3)
print ("g1: ", g1, " g3: ", g3)
print


# Calculate new As
a1 = 1. * g3 / (f1 * g3 - f3 * g1)
a3 = -1. * g1 / (f1 * g3 - f3 * g1)

print ("New a1: ", a1, " New a3: ", a3)
print


# Recalculate p scalars
p1 = (a1 * tripleProduct(R1, pHat2, pHat3) - tripleProduct(R2, pHat2, pHat3)
      + a3 * tripleProduct(R3, pHat2, pHat3)) / (a1 * tripleProduct(pHat1, pHat2, pHat3))
p2 = (a1 * tripleProduct(pHat1, R1, pHat3) - tripleProduct(pHat1, R2, pHat3)
      + a3 * tripleProduct(pHat1, R3, pHat3)) / (-1 * tripleProduct(pHat1, pHat2, pHat3))
p3 = (a1 * tripleProduct(pHat2, R1, pHat1) - tripleProduct(pHat2, R2, pHat1)
      + a3 * tripleProduct(pHat2, R3, pHat1)) / (a3 * tripleProduct(pHat2, pHat3, pHat1))

print ("New rho scalars: ", p1, p2, p3)
print


# Recalculate r vectors
r1 = rCalculate(p1, pHat1, R1)
r2 = rCalculate(p2, pHat2, R2)
r3 = rCalculate(p3, pHat3, R3)
rList = [r1, r3]
r2magnitude = linalg.norm(r2)

print ("New r1: ", r1)
print ("New r2: ", r2)
print ("New r3: ", r3)
print ("New r2 magnitude: ", r2magnitude)
print


# Recalculate r0Dot
r2Dot = [(r3[0] - r1[0]) / tau2, (r3[1] - r1[1]) / tau2, (r3[2] - r1[2]) / tau2]


# Calculate new f and g series
f1 = 1 - 1. / (2 * r2magnitude**3) * tau1**2 + dotProduct(r2, r2Dot) / (2 * r2magnitude**5) * \
    tau1**3 + 1. / 24 * (3. / r2magnitude**3 * (dotProduct(r2Dot, r2Dot) / r2magnitude**2 - 1.
    / r2magnitude**3) - 15 * dotProduct(r2, r2Dot)**2 / r2magnitude**7 + 1. / r2magnitude**6) * tau1**4
f3 = 1 - 1. / (2 * r2magnitude**3) * tau3**2 + dotProduct(r2, r2Dot) / (2 * r2magnitude**5) * \
    tau3**3 + 1. / 24 * (3. / r2magnitude**3 * (dotProduct(r2Dot, r2Dot) / r2magnitude**2 - 1.
    / r2magnitude**3) - 15 * dotProduct(r2, r2Dot)**2 / r2magnitude**7 + 1. / r2magnitude**6) * tau3**4
g1 = tau1 - 1. / (6 * r2magnitude**3) * tau1**3 + dotProduct(r2, r2Dot) / (4 * r2magnitude**5) * tau1**4
g3 = tau3 - 1. / (6 * r2magnitude**3) * tau3**3 + dotProduct(r2, r2Dot) / (4 * r2magnitude**5) * tau3**4

print ("New f1: ", f1, " New f3: ", f3)
print ("New g1: ", g1, " New g3: ", g3)
print


"""
Loop
"""
newr2magnitude = 0
cont = True
while cont:
    oldr2magnitude = newr2magnitude
    # Find values for f and g series
    f1 = 1 - 1. / (2 * r2magnitude**3) * tau1**2 + dotProduct(r2, r2Dot) / (2 * r2magnitude**5) \
        * tau1**3 + 1. / 24 * (3. / r2magnitude**3 * (dotProduct(r2Dot, r2Dot) / r2magnitude**2
        - 1. / r2magnitude**3) - 15 * dotProduct(r2, r2Dot)**2 / r2magnitude**7 + 1. / r2magnitude**6) * tau1**4
    f3 = 1 - 1. / (2 * r2magnitude**3) * tau3**2 + dotProduct(r2, r2Dot) / (2 * r2magnitude**5) \
        * tau3**3 + 1. / 24 * (3. / r2magnitude**3 * (dotProduct(r2Dot, r2Dot) / r2magnitude**2
        - 1. / r2magnitude**3) - 15 * dotProduct(r2, r2Dot)**2 / r2magnitude**7 + 1. / r2magnitude**6) * tau3**4
    g1 = tau1 - 1. / (6 * r2magnitude**3) * tau1**3 + dotProduct(r2, r2Dot) / (4 * r2magnitude**5) * tau1**4
    g3 = tau3 - 1. / (6 * r2magnitude**3) * tau3**3 + dotProduct(r2, r2Dot) / (4 * r2magnitude**5) * tau3**4

    # Determine new values of a1 and a3
    a1 = g3 / (f1 * g3 - f3 * g1)
    a3 = -1. * g1 / (f1 * g3 - f3 * g1)

    # Determine new rho scalars
    p1 = (a1 * tripleProduct(R1, pHat2, pHat3) - tripleProduct(R2, pHat2, pHat3)
          + a3 * tripleProduct(R3, pHat2, pHat3)) / (a1 * tripleProduct(pHat1, pHat2, pHat3))
    p2 = (a1 * tripleProduct(pHat1, R1, pHat3) - tripleProduct(pHat1, R2, pHat3)
          + a3 * tripleProduct(pHat1, R3, pHat3)) / (-1 * tripleProduct(pHat1, pHat2, pHat3))
    p3 = (a1 * tripleProduct(pHat2, R1, pHat1) - tripleProduct(pHat2, R2, pHat1)
          + a3 * tripleProduct(pHat2, R3, pHat1)) / (a3 * tripleProduct(pHat2, pHat3, pHat1))

    # Recalculate r and rDot
    r1 = rCalculate(p1, pHat1, R1)
    r2 = rCalculate(p2, pHat2, R2)
    r3 = rCalculate(p3, pHat3, R3)
    rList = [r1, r3]
    r2magnitude = linalg.norm(r2)
    r2Dot = [f3 * r1[0] / (g1 * f3 - g3 * f1) - f1 * r3[0] / (g1 * f3 - g3 * f1), f3 * r1[1]
             / (g1 * f3 - g3 * f1) - f1 * r3[1] / (g1 * f3 - g3 * f1), f3 * r1[2] / (g1 * f3
             - g3 * f1) - f1 * r3[2] / (g1 * f3 - g3 * f1)]
    newr2magnitude = r2magnitude

    # Light travel time correction
    t1 = t1 - p1 / c
    t2 = t2 - p2 / c
    t3 = t3 - p3 / c
    tau1 = k * (t1 - t2)
    tau2 = k * (t3 - t1)
    tau3 = k * (t3 - t2)

    # Calculate difference between old and new value
    if abs(oldr2magnitude - newr2magnitude) < 1e-3:
        cont = False

print ("r2 (AU, equatorial): ", r2)
print ("r2 dot (AU, equatorial): ", r2Dot)
print
print ("r2 dot (converted to AU/d to match ephemeris): ", [r2Dot[0] * (2 * pi) / 365, r2Dot[1] * (2 * pi) / 365, r2Dot[2] * (2 * pi) / 356])

# Rotate two vectors into ecliptic coordinates
r2 = equatorialtoEcliptic(r2)
r2Dot = equatorialtoEcliptic(r2Dot)

print ("r2 (AU, ecliptic): ", r2)
print ("r2 dot (AU, ecliptic): ", r2Dot)
print


"""
ORBITAL ELEMENTS
"""


def quadrantCheck(sint, cost):
    if sint > 0 and cost > 0:
        return arcsin(sint)
    elif sint > 0 and cost < 0:
        theta = arcsin(sint)
        return pi - theta
    elif sint < 0 and cost < 0:
        theta = -1 * sint
        return arcsin(theta)
        theta = pi + theta
    else:
        theta = arccos(cost)
        return 2 * pi - theta


def vectorRotate(v, axis, theta):
    magnitude = sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    axis = [axis[0] / magnitude, axis[1] / magnitude, axis[2] / magnitude]
    a = cos(theta / 2)
    b = axis[0] * sin(theta / 2.)
    c = axis[1] * sin(theta / 2.)
    d = axis[2] * sin(theta / 2.)

    rotateMatrix = array([[a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                          [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
                          [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - b**2 - c**2]])

    rotatedMatrix = dot(rotateMatrix, v)

    return rotatedMatrix


def angMomentumPerMass(r, rDot):
    au2permodday = (149597871.)**2 * (2 * pi / 365.) * (1 / 24.) * (1 / 3600.)
    hAU = vcrossProduct(r, rDot)
    magHAU = sqrt(hAU[0]**2 + hAU[1]**2 + hAU[2]**2)
    # hKM = [hAU[0]*au2permodday, hAU[1]*au2permodday, hAU[2]*au2permodday]
    # magHKM = magHAU * au2permodday
    # print "h vector (AU^2/modified day): "
    # print hAU
    # print
    # print "h scalar (AU^2/modified day): " + str(magHAU)
    return hAU


def semimajor(r, rDot):
    mu = 1
    magR = linalg.norm(r)
    magrDot = linalg.norm(rDot)
    aAU = 1 / ((2 / magR - magrDot**2 / mu))
    aKM = aAU * 149597871
    print ("a (AU): " + str(aAU))
    # print "a (km): " + str(aKM)
    return aAU


def eccentricity(h, a):
    magHAU = linalg.norm(h)
    e = sqrt(1 - (magHAU**2 / a))
    print ("e: " + str(e))
    return e


def inclination(h):
    hx = h[0]
    hy = h[1]
    hz = h[2]
    i = arctan(sqrt(hx**2 + hy**2) / hz)
    i = i * 180 / pi
    print ("i (degrees): " + str(i))
    return i


def longAscendingNode(i, h):
    hx = h[0]
    hy = h[1]
    magH = linalg.norm(h)
    cosOmega = -hy / (magH * sin(i * pi / 180))
    sinOmega = hx / (magH * sin(i * pi / 180))
    lOmega = quadrantCheck(sinOmega, cosOmega)
    lOmega = lOmega * 180 / pi
    print ("Longitude of the ascending node (degrees): " + str(lOmega))
    return lOmega, sinOmega, cosOmega


def trueLongitude(sinOmega, cosOmega, r, i):
    rx = r[0]
    ry = r[1]
    rz = r[2]
    magR = linalg.norm(r)
    cosU = (rx * cosOmega + ry * sinOmega) / magR
    sinU = rz / (magR * sin(i * pi / 180))
    trueLongitude = quadrantCheck(sinU, cosU)
    trueLongitude = trueLongitude * 180 / pi
    print ("True longitude (degrees): " + str(trueLongitude))
    return trueLongitude


def trueAnomaly(e, h, a, r, rDot):
    magR = linalg.norm(r)
    magH = linalg.norm(h)
    rProduct = dotProduct(r, rDot)
    sinv = a * (1 - e**2) / (e * magH) * rProduct / magR
    cosv = 1 / e * (a * (1 - e**2) / magR - 1)
    v = quadrantCheck(sinv, cosv)
    v = v * 180 / pi
    print ("True anomaly (degrees): " + str(v))
    return v


def argPerihelion(U, v):
    aPerihelion = (U - v) % 360
    print ("Argument of perihelion (in degrees): " + str(aPerihelion))
    return aPerihelion


def meanAnomaly(r, a, e, v):
    cosE = 1 / e * (1 - linalg.norm(r) / a)
    if v < pi:
        # E is between 0 and pi
        if cosE > 0:
            # Quadrant 1
            E = arccos(cosE)
        else:
            # Quadrant 2 - cosE is negative!
            E = pi - arccos(-cosE)
    else:
        # E is between pi and 360
        if cosE > 0:
            # Quadrant 4
            E = 2 * pi - arccos(cosE)
        else:
            # Quadrant 3
            E = pi + arccos(-cosE)
    print (E)
    M = E - e * sin(E)
    M = M * 180 / pi
    print ("Mean Anomaly: " + str(M))
    print
    return M


# Redeclare variables to match format in homework question code
r = r2
rDot = r2Dot


# Running the things and printing the things
print ("################################################################")
print ("Orbital Elements")
h = angMomentumPerMass(r, rDot)
a = semimajor(r, rDot)
e = eccentricity(h, a)
i = inclination(h)
lOmega = (longAscendingNode(i, h))
U = trueLongitude(lOmega[1], lOmega[2], r, i)
v = trueAnomaly(e, h, a, r, rDot)
aPerihelion = argPerihelion(U, v)
M = meanAnomaly(r, a, e, v)


# # Do percent differences with JPL values
# # NOTE omega = aPerihelion
# aJPL = 1.820221560664795
# eJPL = 0.3264825227417676
# iJPL = 23.25678573253114
# lOmegaJPL = 164.85231665456
# omegaJPL = 154.3654325836887
# MJPL = 3.517493888576719e2


# # Percent difference calculator
# def perDiff(var1, var2):
#     diff = abs(var1 - var2) / ((var1 + var2) / 2) * 100
#     return diff

# print ("################################################################")
# print ("Percent Differences (as compared to JPL values)")
# print ("a: " + str(perDiff(a, aJPL)) + "%")
# print ("e: " + str(perDiff(e, eJPL)) + "%")
# print ("i: " + str(perDiff(i, iJPL)) + "%")
# print ("Lon ascending node: " + str(perDiff(lOmega[0], lOmegaJPL)) + "%")
# print ("Perihelion: " + str(perDiff(aPerihelion, omegaJPL)) + "%")
# print ("M: " + str(perDiff(M, MJPL)) + "%")

a = a * units.AU
ecc = e * units.one
inc = i * units.deg
raan = lOmega[0] * units.deg
argp = aPerihelion * units.deg
nu = v * units.deg
epoch = time.Time(t2,format='jd',scale='utc')

earth_orbit = Orbit.from_body_ephem(Earth)
earth_orbit = earth_orbit.propagate(time.Time(t2,format='jd',scale='tdb'),rtol=1e-10)
magellan_orbit = neows.orbit_from_name('2018ez2')
magellan_orbit = magellan_orbit.propagate(time.Time(t2,format='jd',scale='tdb'),rtol=1e-10)
estimated_orbit = Orbit.from_classical(Sun, a, ecc, inc, raan, argp, nu, epoch)

op = OrbitPlotter()
op.plot(earth_orbit, label='Earth')
op.plot(magellan_orbit, label='2018 EZ2')
op.plot(estimated_orbit, label='Estimated')

plt.show()