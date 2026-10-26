"""Wave-frequency vessel motion from Motion RAOs and wave components.

License: GPL-3.0-or-later (see LICENSE and NOTICE.md in this folder)
Authors: Araz, Olve Grønås Birkeland (2026)
"""

import numpy as np

if __package__:
    from .LF import kinematics
    from .WaveLoads import WaveComponents, check_pose_time, spatial_phase
else:
    from LF import kinematics
    from WaveLoads import WaveComponents, check_pose_time, spatial_phase


class WF:
    def __init__(self, motion_table, *, gravity=9.81):
        # INPUT from Wamit.py: result["motion_table"], six Motion RAOs (not Force RAOs).
        # INPUT from Wamit.py: result["gravity"] [m/s^2].
        if motion_table.kind != 'rao' or motion_table.values.shape[0] != 6:
            raise ValueError('WF needs six motion-RAO rows, not force RAOs')
        if not np.isfinite(gravity) or gravity <= 0:
            raise ValueError('gravity must be positive')
        self.table, self.gravity = motion_table, gravity
        self.set_components(WaveComponents([], [], [], []))

    def set_components(self, components):
        # INPUT from waves via hull/vessel.py:
        # amplitude [m], omega [rad/s], phase [rad], direction [rad].
        if not isinstance(components, WaveComponents):
            raise TypeError('Expected WaveComponents')
        self.components = components
        self._motion = self.table.at_frequencies(components.omega)

    def displacement(self, time, eta_lf):
        # INPUT: simulation time [s] and LF pose [north,east,down,roll,pitch,yaw].
        eta = check_pose_time(time, eta_lf)
        w = self.components

        # Find the Motion RAO H at each wave frequency and relative heading.
        H = self.table.at_headings(self._motion, w.direction-eta[5])
        phase = spatial_phase(w, time, eta, self.gravity)

        # FORMULA: body_motion = Re{conj(H) @ [amplitude * exp(i * phase)]}.
        # Motion RAOs already describe first-order wave motion; do not add first-order
        # forces to LF.
        body_motion = np.real(np.conj(H) @ (w.amplitude*np.exp(1j*phase)))

        # Convert body-frame motion to NED displacement and Euler-angle increments.
        wf_displacement = kinematics(eta) @ body_motion
        if not np.all(np.isfinite(wf_displacement)):
            raise ValueError('WF calculation produced nonfinite motion')

        # OUTPUT: [north,east,down,roll,pitch,yaw] WF increments (m/rad).
        # hull/vessel.py adds these to the LF pose for the total vessel pose.
        return wf_displacement
