import itertools
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized_class
from openpilot.cereal import log
from openpilot.selfdrive.car.cruise import VCruiseHelper, V_CRUISE_MIN, V_CRUISE_MAX, V_CRUISE_INITIAL, IMPERIAL_INCREMENT
from openpilot.cereal import custom
from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


def run_cruise_simulation(cruise, personality, t_end=20.):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=max(cruise - 1., 0.0),
    lead_relevancy=True,
    initial_distance_lead=100,
    cruise_values=[cruise],
    prob_lead_values=[0.0],
    breakpoints=[0.],
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1, 3]


@parameterized_class(("personality", "speed"), itertools.product(
                      log.LongitudinalPersonality.schema.enumerants, # personality
                      [5,35])) # speed
class TestCruiseSpeed(OpenpilotTestCase):
  def test_cruise_speed(self):
    print(f'Testing {self.speed} m/s')
    cruise_speed = float(self.speed)

    simulation_steady_state = run_cruise_simulation(cruise_speed, self.personality)
    self.assertAlmostEqual(simulation_steady_state, cruise_speed, delta=.01, msg=f'Did not reach {self.speed} m/s')


# TODO: test pcmCruise and pcmCruiseSpeed
@parameterized_class(('pcm_cruise', 'pcm_cruise_speed'), [(False, True)])
class TestVCruiseHelper(OpenpilotTestCase):
  def setup_method(self):
    self.CP = car.CarParams(pcmCruise=self.pcm_cruise)
    self.CP_SP = custom.CarParamsSP(pcmCruiseSpeed=self.pcm_cruise_speed)
    self.v_cruise_helper = VCruiseHelper(self.CP, self.CP_SP)
    self.reset_cruise_speed_state()

  def reset_cruise_speed_state(self):
    # Two resets previous cruise speed
    for _ in range(2):
      self.v_cruise_helper.update_v_cruise(car.CarState(cruiseState={"available": False}), enabled=False, is_metric=False)

  def enable(self, v_ego):
    # Simulates user pressing set with a current speed
    self.v_cruise_helper.initialize_v_cruise(car.CarState(vEgo=v_ego), V_CRUISE_INITIAL)

  def test_adjust_speed(self):
    """
    Asserts speed changes on falling edges of buttons.
    """

    self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS)

    for btn in (ButtonType.accelCruise, ButtonType.decelCruise):
      for pressed in (True, False):
        CS = car.CarState(cruiseState={"available": True})
        CS.buttonEvents = [ButtonEvent(type=btn, pressed=pressed)]

        self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)
        assert pressed == (self.v_cruise_helper.v_cruise_kph == self.v_cruise_helper.v_cruise_kph_last)

  def test_rising_edge_enable(self):
    """
    Some car interfaces may enable on rising edge of a button,
    ensure we don't adjust speed if enabled changes mid-press.
    """

    # NOTE: enabled is always one frame behind the result from button press in controlsd
    for enabled, pressed in ((False, False),
                             (False, True),
                             (True, False)):
      CS = car.CarState(cruiseState={"available": True})
      CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=pressed)]
      self.v_cruise_helper.update_v_cruise(CS, enabled=enabled, is_metric=False)
      if pressed:
        self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS)

      # Expected diff on enabling. Speed should not change on falling edge of pressed
      assert (not pressed) == (self.v_cruise_helper.v_cruise_kph == self.v_cruise_helper.v_cruise_kph_last)

  def test_resume_in_standstill(self):
    """
    Asserts we don't increment set speed if user presses resume/accel to exit cruise standstill.
    """

    self.enable(0)

    for standstill in (True, False):
      for pressed in (True, False):
        CS = car.CarState(cruiseState={"available": True, "standstill": standstill})
        CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=pressed)]
        self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)

        # speed should only update if not at standstill and button falling edge
        should_equal = standstill or pressed
        assert should_equal == (self.v_cruise_helper.v_cruise_kph == self.v_cruise_helper.v_cruise_kph_last)

  def test_set_gas_pressed(self):
    """
    Asserts pressing set while enabled with gas pressed sets
    the speed to the maximum of vEgo and current cruise speed.
    """

    for v_ego in np.linspace(0, 100, 101):
      self.reset_cruise_speed_state()
      self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS)

      # first decrement speed, then perform gas pressed logic
      expected_v_cruise_kph = self.v_cruise_helper.v_cruise_kph - IMPERIAL_INCREMENT
      expected_v_cruise_kph = max(expected_v_cruise_kph, v_ego * CV.MS_TO_KPH)  # clip to min of vEgo
      expected_v_cruise_kph = float(np.clip(round(expected_v_cruise_kph, 1), V_CRUISE_MIN, V_CRUISE_MAX))

      CS = car.CarState(vEgo=float(v_ego), gasPressed=True, cruiseState={"available": True})
      CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
      self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)

      # TODO: fix skipping first run due to enabled on rising edge exception
      if v_ego == 0.0:
        continue
      assert expected_v_cruise_kph == self.v_cruise_helper.v_cruise_kph

  def test_initialize_v_cruise(self):
    """
    Asserts allowed cruise speeds on enabling with SET.
    """

    for v_ego in np.linspace(0, 100, 101):
      self.reset_cruise_speed_state()
      assert not self.v_cruise_helper.v_cruise_initialized

      self.enable(float(v_ego))
      assert V_CRUISE_INITIAL <= self.v_cruise_helper.v_cruise_kph <= V_CRUISE_MAX
      assert self.v_cruise_helper.v_cruise_initialized


class TestVCruiseHelperDashScale:
  """KIA_CEED with openpilot long: set speed on the dash (speedometer) scale, planner gets true speed."""

  @staticmethod
  def helper(fingerprint, pcm_cruise=False):
    return VCruiseHelper(car.CarParams(carFingerprint=fingerprint, pcmCruise=pcm_cruise), custom.CarParamsSP(pcmCruiseSpeed=True))

  def test_engage_at_dash_120(self):
    h = self.helper("KIA_CEED")
    assert h.v_cruise_planner_kph == 255  # UNSET passes through
    h.initialize_v_cruise(car.CarState(vEgo=float(114.3 * CV.KPH_TO_MS), vEgoCluster=float(120 * CV.KPH_TO_MS)), V_CRUISE_INITIAL)
    assert h.v_cruise_kph == 120
    assert h.v_cruise_cluster_kph == 120
    assert abs(h.v_cruise_planner_kph - (120 - 3.68) / 1.0216) < 1e-6  # ~113.86 km/h true speed
    assert abs(h.v_cruise_planner_kph - 114.3) < 0.5

  def test_missing_cluster_speed_falls_back_to_v_ego(self):
    h = self.helper("KIA_CEED")
    h.initialize_v_cruise(car.CarState(vEgo=float(100 * CV.KPH_TO_MS)), V_CRUISE_INITIAL)
    assert h.v_cruise_kph == 100

  def test_other_car_and_stock_long_unchanged(self):
    for h in (self.helper("HYUNDAI_SONATA"), self.helper("KIA_CEED", pcm_cruise=True)):
      assert not h.dash_scale
      h.CP = car.CarParams(carFingerprint=h.CP.carFingerprint, pcmCruise=False)  # let initialize_v_cruise run
      h.initialize_v_cruise(car.CarState(vEgo=float(114.3 * CV.KPH_TO_MS), vEgoCluster=float(120 * CV.KPH_TO_MS)), V_CRUISE_INITIAL)
      assert h.v_cruise_kph == 114
      assert h.v_cruise_planner_kph == h.v_cruise_kph == h.v_cruise_cluster_kph

  def test_gas_override_set_clips_to_dash_speed(self):
    h = self.helper("KIA_CEED")
    h.initialize_v_cruise(car.CarState(vEgo=float(80 * CV.KPH_TO_MS), vEgoCluster=float(85 * CV.KPH_TO_MS)), V_CRUISE_INITIAL)
    h.update_v_cruise(car.CarState(cruiseState={"available": True}), enabled=True, is_metric=True)  # rising-edge enable frame
    for pressed in (True, False):
      CS = car.CarState(vEgo=float(114.3 * CV.KPH_TO_MS), vEgoCluster=float(120 * CV.KPH_TO_MS), gasPressed=True, cruiseState={"available": True})
      CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=pressed)]
      h.update_v_cruise(CS, enabled=True, is_metric=True)
    assert h.v_cruise_kph == 120
    assert h.v_cruise_cluster_kph == 120

  def test_sentinels_pass_through(self):
    from openpilot.selfdrive.car.cruise import dash_to_planner_kph, V_CRUISE_UNSET
    for v in (V_CRUISE_UNSET, -1, 0):
      assert dash_to_planner_kph(v) == v
    assert dash_to_planner_kph(V_CRUISE_MAX) < V_CRUISE_MAX  # dash 180 -> ~172.6 true
