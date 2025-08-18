# kalman_aircraft.py
import math
import numpy as np

KT_TO_MPS = 0.514444
FT_TO_M = 0.3048
FPM_TO_MPS = 0.00508

def track_deg_to_vxvy(speed_kt, track_deg):
    if speed_kt is None or track_deg is None:
        return None, None
    v = float(speed_kt) * KT_TO_MPS
    th = math.radians(float(track_deg))
    # aviation "track": 0=N, 90=E -> vx=east, vy=north
    vx = v * math.sin(th)
    vy = v * math.cos(th)
    return vx, vy

def vxvy_to_speed_track(vx, vy):
    v = math.hypot(vx, vy)
    # 0=N, 90=E -> atan2(East, North)
    th = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0
    return v, th

class AircraftKF:
    """
    Minimal NCV Kalman filter:
      state x = [x, y, z, vx, vy, vz]^T   (SI units)
    Measurements we can use: z, vz, vx, vy (from your ADS-B fields).
    """
    def __init__(
        self,
        q_h=0.25,      # horiz accel noise var (m^2/s^4) ~ (0.5 m/s^2)^2
        q_v=0.25,      # vertical accel noise var
        r_alt=15.0,    # altitude meas std (m) ~ 50 ft
        r_vxy=2.0,     # vx,vy meas std (m/s) ~ from speed/track noise
        r_vz=0.5,      # vertical rate meas std (m/s) ~ 100 fpm
        init_pos_std=5e3,  # large pos std (m) to not bias solution
        init_vel_std=150.0 # large vel std (m/s)
    ):
        self.q_h = float(q_h)
        self.q_v = float(q_v)
        self.r_alt = float(r_alt)
        self.r_vxy = float(r_vxy)
        self.r_vz = float(r_vz)
        self.x = None   # 6x1
        self.P = None   # 6x6
        self.t = None   # last timestamp (seconds)

        self.init_pos_var = init_pos_std**2
        self.init_vel_var = init_vel_std**2

    def _F_and_Q(self, dt):
        F = np.eye(6)
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt

        # NCV 1D block Q(dt) = q * [[dt^3/3, dt^2/2],[dt^2/2, dt]]
        dt2 = dt*dt
        dt3 = dt2*dt
        Q = np.zeros((6,6))
        # x-vx block
        qx = self.q_h
        Q[0,0] = qx * (dt3/3); Q[0,3] = qx * (dt2/2)
        Q[3,0] = qx * (dt2/2); Q[3,3] = qx *  dt
        # y-vy block
        Q[1,1] = qx * (dt3/3); Q[1,4] = qx * (dt2/2)
        Q[4,1] = qx * (dt2/2); Q[4,4] = qx *  dt
        # z-vz block
        qz = self.q_v
        Q[2,2] = qz * (dt3/3); Q[2,5] = qz * (dt2/2)
        Q[5,2] = qz * (dt2/2); Q[5,5] = qz *  dt

        return F, Q

    def _ensure_initialized(self, t, altitude_ft=None, speed_kt=None, track_deg=None, vert_rate_fpm=None):
        # Build an initial state from whatever we have
        if self.x is not None:
            return
        z = (altitude_ft or 0.0) * FT_TO_M
        vz = (vert_rate_fpm or 0.0) * FPM_TO_MPS
        vx, vy = track_deg_to_vxvy(speed_kt, track_deg)
        vx = 0.0 if vx is None else vx
        vy = 0.0 if vy is None else vy

        self.x = np.array([[0.0],[0.0],[z],[vx],[vy],[vz]], dtype=float)
        self.P = np.diag([self.init_pos_var, self.init_pos_var, self.init_pos_var,
                          self.init_vel_var, self.init_vel_var, self.init_vel_var])
        self.t = float(t)

    def predict_to(self, t_new):
        """Propagate to time t_new (seconds)."""
        if self.x is None:
            return
        dt = max(0.0, float(t_new) - self.t)
        if dt == 0.0:
            return
        F, Q = self._F_and_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t_new)

    def _update_linear(self, H, z, R):
        # Standard KF update
        y = z - H @ self.x                # innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    def update(self, t_meas, altitude_ft=None, speed_kt=None, track_deg=None, vert_rate_fpm=None):
        """
        Accept any subset of (altitude, speed+track, vertical rate) at time t_meas (seconds).
        """
        self._ensure_initialized(t_meas, altitude_ft, speed_kt, track_deg, vert_rate_fpm)
        self.predict_to(t_meas)

        # Stack available measurements sequentially (each small linear update)
        # 1) altitude -> measure z
        if altitude_ft is not None:
            z_meas = np.array([[altitude_ft * FT_TO_M]], dtype=float)
            H = np.zeros((1,6)); H[0,2] = 1.0    # measure z
            R = np.array([[self.r_alt**2]], dtype=float)
            self._update_linear(H, z_meas, R)

        # 2) vertical rate -> measure vz
        if vert_rate_fpm is not None:
            vz_meas = np.array([[vert_rate_fpm * FPM_TO_MPS]], dtype=float)
            H = np.zeros((1,6)); H[0,5] = 1.0    # measure vz
            R = np.array([[self.r_vz**2]], dtype=float)
            self._update_linear(H, vz_meas, R)

        # 3) speed+track -> measure (vx, vy)
        if (speed_kt is not None) and (track_deg is not None):
            vx_m, vy_m = track_deg_to_vxvy(speed_kt, track_deg)
            if (vx_m is not None) and (vy_m is not None):
                z_meas = np.array([[vx_m],[vy_m]], dtype=float)
                H = np.zeros((2,6)); H[0,3] = 1.0; H[1,4] = 1.0  # measure vx, vy
                R = np.diag([self.r_vxy**2, self.r_vxy**2])
                self._update_linear(H, z_meas, R)

    def predict_next(self, dt_ahead=10.0):
        """
        Predict forward by dt_ahead seconds and return (altitude_ft, speed_kt, track_deg).
        Non-destructive: does not advance internal time.
        """
        if self.x is None:
            return None, None, None
        # Copy state
        x = self.x.copy(); P = self.P.copy(); t = self.t
        F, Q = self._F_and_Q(dt_ahead)
        x_pred = F @ x
        vz = float(x_pred[5,0])
        z  = float(x_pred[2,0])
        vx = float(x_pred[3,0])
        vy = float(x_pred[4,0])
        v, track = vxvy_to_speed_track(vx, vy)
        alt_ft = z / FT_TO_M
        spd_kt = v / KT_TO_MPS
        # (We don't overwrite filter state)
        return alt_ft, spd_kt, track