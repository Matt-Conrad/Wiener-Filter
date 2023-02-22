# CGM-Weiner-Filter
This repository is meant to house the implementation of "Interstitial fluid glucose time-lag correction for real-time continuous glucose monitoring" by Keenan et al.

The purpose of this repository is to explore the domain of glucose monitoring and to review the interesting concept of the Weiner filter from signal processing theory.

## Signal Generation

In the paper, the authors how the process of glucose monitoring can be represented as a simple model as seen below:

<img src="images/physioModel.png" alt="drawing" width="750"/>

This model is essentially a differential equation where glucose shifts according to concentrations in the 3 compartments: capillary, interstitial fluid (ISF), and fat/muscle. The capillary compartment is important because that's what we're interested in finding the blood glucose (BG) level of, and the ISF compartment is important because that's where the measuring by the sensor occurs; the fat/muscle compartment is not as important and only matters for the clearing of the ISF. 

This physio model can be represented as a signal processing diagram below 

<img src="images/blockDiagram.png" alt="drawing" width="750"/>

In this diagram, s(n) is the BG of the capillary compartment, H(z) is a filter representing the barrier between capillary and ISF, y(n) is the BG of the ISF compartment, e(n) is electronic noise from the sensor/hardware, x(n) is the digitized signal, G(z) is the Wiener filter, and s'(n) is the estimate for the capillary BG. 

In order to create the capillary BG signal, I relied on a simulator called simglucose that generates somewhat realistic BG signals with customizability without the need for the actual data, which may be hard to come by. 

Once the BG signal is created, the next step is to pass it through the diffusion filter. The authors outline that they used the following: 

<img src="images/diffusionFilterTransferFxn.png" alt="drawing" width="750"/>

The diffusion filter was derived from a differential equation modeling the change in the ISF's total glucose over time. While I will not go through the entire derivation, however the differential equation is here:

<img src="images/diffEq.png" alt="drawing" width="750"/>

The diffusion filter values are characterized by a time lag (tau) of 10 minutes, a 1-minute sample interval, and a unity gain.

What is a unity gain? Well, in this model the sensor signal (x(n)) is a scalar ($\alpha$) multiple of y(n), and y(n) is scalar ($\Kappa$) multiple of capillary BG (s(n)). Together, these scalar multiples create a non-unity gain between s(n) and x(n) and must be corrected using a calibration factor (1/($\alpha\Kappa$)) to make it a unity gain. This calibration factor is often calculated using finger sticks to get s(n) in order to calibrate x(n). By using the H(z) from above, we're essentially assuming the sensor we're using is already calibrated.  



## Citations
1. Jinyu Xie. Simglucose v0.2.1 (2018) [Online]. Available: https://github.com/jxx123/simglucose. Accessed on: August-15-2022.
2. Irina Gaynanova. Awesome-CGM v1.1.0 (2021) [Online]. Available: https://github.com/irinagain/Awesome-CGM. Accessed: September-27-2022.
