from xgolib import XGO
import time
import math


# ============================================================
# XGO INITIALIZATION
# ============================================================

dog = XGO(port='/dev/ttyAMA0', version='xgolite')
dog.motor_speed(50)


# ============================================================
# CPG PARAMETERS
# ============================================================

# Integration step for the neural simulation
DT = 0.1

# Real-time delay between CPG updates.
#
# 0.001 gives roughly a 1.8 second cycle once the CPG
# reaches its steady rhythm.
#
# If the robot walks too quickly, increase this to 0.0015
# or 0.002.
REAL_DT = 0.001

# Membrane capacitance
C_MEM = 1.0

# Time constants
TAU_F = 1.0
TAU_S = 50.0
TAU_US = 2500.0

# Bio-mimetic neuron parameters
ALPHA_F = -2.0
ALPHA_SP = 2.0
ALPHA_SN = -1.5
ALPHA_US = 1.5

# External currents
#
# FL and BR receive a small negative bias.
# FR and BL receive zero bias.
#
# This creates the phase difference between the
# two diagonal leg groups.
BASE_IAPP = 0.0
PHASE_BIAS = -0.10

IAPP = [
    BASE_IAPP + PHASE_BIAS,  # FL
    BASE_IAPP,               # FR
    BASE_IAPP + PHASE_BIAS,  # BR
    BASE_IAPP                # BL
]

# Synaptic coupling
W = -0.20
W_RETURN = 0.90 * W

# Activity function used for synaptic coupling
ACTIVITY_OFFSET = 0.5

# Burst/spike threshold
BURST_THRESHOLD = 1.5


# ============================================================
# XGO LEG / GAIT PARAMETERS
# ============================================================

# XGO leg IDs:
#
# 1 = Front Left
# 2 = Front Right
# 3 = Back Right
# 4 = Back Left
#
# These correspond to:
# CPG neuron 0 = FL
# CPG neuron 1 = FR
# CPG neuron 2 = BR
# CPG neuron 3 = BL

LEG_IDS = [1, 2, 3, 4]

LEG_NAMES = [
    "FL",
    "FR",
    "BR",
    "BL"
]

# Foot coordinates are in millimeters.
#
# +X = forward
# -X = backward
# +Z = downward
#
# Therefore a smaller Z lifts the foot.

FORWARD_X = 20
BACK_X = -20

GROUND_Z = 100
LIFT_Z = 82

Y_POSITION = 0


# ============================================================
# GAIT TIMING
# ============================================================

# When a CPG neuron bursts:
#
#   0.00 s   Lift foot while it is behind the body
#   0.10 s   Swing foot forward while lifted
#   0.22 s   Put foot down
#   0.45 s   Move foot backward while on ground
#
# The backward motion against the ground is what pushes
# the robot's body forward.

SWING_FORWARD_TIME = 0.10
PUT_DOWN_TIME = 0.22
PUSH_TIME = 0.45


# ============================================================
# BIO-MIMETIC NEURON
# ============================================================

class BioNeuron:

    def __init__(self, iapp):

        self.Iapp = iapp

        # Main membrane voltage
        self.Vm = 0.0

        # Fast state
        self.Vf = 0.0

        # Slow state
        self.Vs = 0.0

        # Ultra-slow state
        self.Vus = 0.0

    def activity(self):

        return math.tanh(
            self.Vm - ACTIVITY_OFFSET
        )

    def calculate_derivatives(self, Isyn):

        # Fast negative current
        If = ALPHA_F * math.tanh(self.Vf)

        # Slow positive current
        Isp = ALPHA_SP * math.tanh(self.Vs)

        # Slow negative current
        Isn = ALPHA_SN * math.tanh(self.Vs)

        # Ultra-slow positive current
        Ius = ALPHA_US * math.tanh(self.Vus)

        # Main membrane equation
        #
        # C dVm/dt =
        # -(Vm + If + Isp + Isn + Ius - Iapp - Isyn)

        dVm = (
            -(
                self.Vm
                + If
                + Isp
                + Isn
                + Ius
                - self.Iapp
                - Isyn
            )
            / C_MEM
        )

        # Fast state
        dVf = (
            self.Vm - self.Vf
        ) / TAU_F

        # Slow state
        dVs = (
            self.Vm - self.Vs
        ) / TAU_S

        # Ultra-slow state
        dVus = (
            self.Vm - self.Vus
        ) / TAU_US

        return dVm, dVf, dVs, dVus

    def update(self, derivatives):

        dVm, dVf, dVs, dVus = derivatives

        self.Vm += DT * dVm
        self.Vf += DT * dVf
        self.Vs += DT * dVs
        self.Vus += DT * dVus


# ============================================================
# CPG NETWORK
# ============================================================

class CPG:

    def __init__(self):

        # Neuron order:
        #
        # 0 = FL
        # 1 = FR
        # 2 = BR
        # 3 = BL

        self.neurons = [
            BioNeuron(IAPP[0]),
            BioNeuron(IAPP[1]),
            BioNeuron(IAPP[2]),
            BioNeuron(IAPP[3])
        ]

        self.previous_vm = [
            neuron.Vm
            for neuron in self.neurons
        ]

    def step(self):

        # ----------------------------------------------------
        # Save old voltages
        # ----------------------------------------------------

        self.previous_vm = [
            neuron.Vm
            for neuron in self.neurons
        ]

        # ----------------------------------------------------
        # Calculate neuron activities
        # ----------------------------------------------------

        activities = [
            neuron.activity()
            for neuron in self.neurons
        ]

        # ----------------------------------------------------
        # Calculate synaptic currents
        # ----------------------------------------------------

        Isyn = [0.0, 0.0, 0.0, 0.0]

        # FR and BL receive inhibition from FL and BR
        #
        # FR <- FL
        # FR <- BR
        # BL <- FL
        # BL <- BR

        Isyn[1] += W * activities[0]
        Isyn[1] += W * activities[2]

        Isyn[3] += W * activities[0]
        Isyn[3] += W * activities[2]

        # FL and BR receive the weaker return inhibition
        #
        # FL <- FR
        # FL <- BL
        # BR <- FR
        # BR <- BL

        Isyn[0] += W_RETURN * activities[1]
        Isyn[0] += W_RETURN * activities[3]

        Isyn[2] += W_RETURN * activities[1]
        Isyn[2] += W_RETURN * activities[3]

        # ----------------------------------------------------
        # Calculate derivatives FIRST
        #
        # This makes all four neurons update from the same
        # previous state.
        # ----------------------------------------------------

        derivatives = []

        for i in range(4):

            d = self.neurons[i].calculate_derivatives(
                Isyn[i]
            )

            derivatives.append(d)

        # ----------------------------------------------------
        # Update all neurons
        # ----------------------------------------------------

        for i in range(4):

            self.neurons[i].update(
                derivatives[i]
            )

        # ----------------------------------------------------
        # Detect threshold crossings
        # ----------------------------------------------------

        bursts = []

        for i in range(4):

            old_vm = self.previous_vm[i]
            new_vm = self.neurons[i].Vm

            if (
                old_vm < BURST_THRESHOLD
                and new_vm >= BURST_THRESHOLD
            ):

                bursts.append(i)

        return bursts


# ============================================================
# GAIT CONTROLLER
# ============================================================

class GaitController:

    def __init__(self, dog):

        self.dog = dog

        # Each element is either:
        #
        # None
        #
        # or:
        # {
        #     "start": time,
        #     "stage": number
        # }
        #
        # One entry per leg.

        self.active = [
            None,
            None,
            None,
            None
        ]

    def set_leg(self, leg_number, x, z):

        # XGO leg() uses:
        #
        # [x, y, z]
        #
        # x = forward/back
        # y = left/right
        # z = up/down

        self.dog.leg(
            leg_number,
            [
                x,
                Y_POSITION,
                z
            ]
        )

    def burst(self, neuron_index):

        leg_number = LEG_IDS[neuron_index]
        leg_name = LEG_NAMES[neuron_index]

        now = time.monotonic()

        # Start a new swing cycle.

        self.active[neuron_index] = {
            "start": now,
            "stage": 0
        }

        print(
            "BURST:",
            leg_name
        )

        # ----------------------------------------------------
        # STAGE 0
        #
        # Lift the foot while it is behind the body.
        # ----------------------------------------------------

        self.set_leg(
            leg_number,
            BACK_X,
            LIFT_Z
        )

    def update(self):

        now = time.monotonic()

        for i in range(4):

            state = self.active[i]

            if state is None:
                continue

            leg_number = LEG_IDS[i]
            start = state["start"]
            stage = state["stage"]

            elapsed = now - start

            # ------------------------------------------------
            # STAGE 1
            #
            # Swing the lifted foot forward.
            # ------------------------------------------------

            if (
                stage == 0
                and elapsed >= SWING_FORWARD_TIME
            ):

                self.set_leg(
                    leg_number,
                    FORWARD_X,
                    LIFT_Z
                )

                state["stage"] = 1

            # ------------------------------------------------
            # STAGE 2
            #
            # Put the foot down in front of the body.
            # ------------------------------------------------

            elif (
                stage == 1
                and elapsed >= PUT_DOWN_TIME
            ):

                self.set_leg(
                    leg_number,
                    FORWARD_X,
                    GROUND_Z
                )

                state["stage"] = 2

            # ------------------------------------------------
            # STAGE 3
            #
            # Push the foot backward against the ground.
            #
            # This is the part that moves the body forward.
            # ------------------------------------------------

            elif (
                stage == 2
                and elapsed >= PUSH_TIME
            ):

                self.set_leg(
                    leg_number,
                    BACK_X,
                    GROUND_Z
                )

                state["stage"] = 3

            # ------------------------------------------------
            # STAGE 4
            #
            # The foot stays in the rear stance position
            # until the next CPG burst.
            # ------------------------------------------------

            elif (
                stage == 3
                and elapsed >= 0.50
            ):

                self.active[i] = None


# ============================================================
# INITIAL ROBOT POSITION
# ============================================================

def initialize_robot():

    print()
    print("Resetting XGO...")

    dog.reset()

    time.sleep(1.0)

    print("Setting starting foot positions...")

    for leg in LEG_IDS:

        dog.leg(
            leg,
            [
                0,
                Y_POSITION,
                GROUND_Z
            ]
        )

        time.sleep(0.1)

    time.sleep(1.0)

    print("Robot ready.")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" XGO Lite Bio-Mimetic CPG Controller")
    print("========================================")
    print()

    initialize_robot()

    cpg = CPG()

    gait = GaitController(dog)

    print("Starting CPG...")
    print("FL + BR and FR + BL should alternate.")
    print()
    print("Press CTRL+C to stop.")
    print()

    try:

        while True:

            # ------------------------------------------------
            # Run one CPG timestep
            # ------------------------------------------------

            bursts = cpg.step()

            # ------------------------------------------------
            # Convert CPG bursts into leg movements
            # ------------------------------------------------

            for neuron_index in bursts:

                gait.burst(
                    neuron_index
                )

            # ------------------------------------------------
            # Advance all active leg trajectories
            # ------------------------------------------------

            gait.update()

            # ------------------------------------------------
            # Optional voltage display
            # ------------------------------------------------

            # Uncomment these lines if you want to see
            # the CPG voltages while it is running.
            #
            # print(
            #     "FL={:+.2f} FR={:+.2f} BR={:+.2f} BL={:+.2f}".format(
            #         cpg.neurons[0].Vm,
            #         cpg.neurons[1].Vm,
            #         cpg.neurons[2].Vm,
            #         cpg.neurons[3].Vm
            #     )
            # )

            # ------------------------------------------------
            # Real-time pacing
            # ------------------------------------------------

            time.sleep(REAL_DT)

    except KeyboardInterrupt:

        print()
        print("Stopping CPG...")

    finally:

        print("Resetting XGO...")

        dog.reset()

        print("Done.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
