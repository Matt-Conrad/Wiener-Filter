from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.simulation.sim_engine import SimObj, batch_sim, sim

from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv

import datetime
import pandas as pd
import copy
import random

def generateBGs(numBGs, bgTime):
    simInstances = createSimInstances(numBGs, bgTime)

    ### Carry out simulation
    results = pd.DataFrame()
    for i, s in enumerate(simInstances[:numBGs]):
        # Note: BG changes when bolus != 0.0 in the stepping
        series = sim(s)
        series = series.rename(columns={'BG': f'BG{i}'})
        if i == 0:
            results = series[[f'BG{i}']]
        else:
            results[f'BG{i}'] = series[[f'BG{i}']]

    # This is for batching the simulations
    # results = batch_sim(sim_instances, parallel=False)

    return results

def createSimInstances(numBGs, bgTime):
    simTime = datetime.timedelta(hours=bgTime)

    patientParams = pd.read_csv('params/vpatient_params.csv')

    patientNameOptions = list(patientParams['Name'].values)
    patientNames = [patientNameOptions[random.randint(0, len(patientNameOptions)-1)] for p in range(numBGs)]

    pumpParams = pd.read_csv('params/pump_params.csv')

    envs = [buildEnv(p, pumpParams) for p in patientNames]

    controller = BBController() # I think this defines the behavior of the system, think control systems
    controllers = [copy.deepcopy(controller) for _ in range(len(envs))]

    # These carry out the simulation
    simInstances = [SimObj(e, c, simTime, animate=False, path="output") for (e, c) in zip(envs, controllers)]

    return simInstances

def buildEnv(pname, pumpParams):
    patient = T1DPatient.withName(pname)

    sensorParamCSV = pd.read_csv('params/sensor_params.csv')
    sensorParams = sensorParamCSV.loc[sensorParamCSV.Name == "Dexcom"].squeeze()
    cgmSensor =  CGMSensor(sensorParams, seed=None)

    params = pumpParams.loc[pumpParams.Name == "Insulet"].squeeze()
    insulinPump = InsulinPump(params)

    day = datetime.date(2000, 1, 1)
    startTime = datetime.datetime.combine(day, datetime.datetime.min.time())
    scenario = RandomScenario(startTime, seed=None)

    scen = copy.deepcopy(scenario)

    return T1DSimEnv(patient, cgmSensor, insulinPump, scen)

