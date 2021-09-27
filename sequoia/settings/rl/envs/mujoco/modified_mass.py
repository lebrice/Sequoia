from gym.envs.mujoco import MujocoEnv
import numpy as np
from typing import Union, List, Dict, ClassVar, TypeVar, Mapping
from functools import partial

V = TypeVar("V")


class ModifiedMassEnv(MujocoEnv):
    """
    Allows the mass of body parts to be changed.

    NOTE: Haven't yet checked how this affects the physics simulation! Might not be 100% working.
    """

    # IDEA: Use somethign like this to tell appart modifications which can be applied
    # on-the-fly on a given env to get multiple tasks, vs those that require creating a
    # new environment for each task.
    CAN_BE_UPDATED_IN_PLACE: ClassVar[bool] = True
    BODY_NAMES: ClassVar[List[str]]

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        body_name_to_mass_scale: Dict[str, float] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            **kwargs,
        )
        body_name_to_mass_scale = body_name_to_mass_scale or {}
        self.default_masses_dict: Dict[str, float] = {
            body_name: self.model.body_mass[i]
            for i, body_name in enumerate(self.model.body_names)
        }
        self.default_masses: np.ndarray = np.copy(self.model.body_mass)

        # dict(zip(body_parts, mass_scales))
        self.scale_masses(**body_name_to_mass_scale)
        # self.model.body_mass = self.get_and_modify_bodymass(body_part, mass_scale)
        # self.model._compute_subtree()
        # self.model.forward()

    def __init_subclass__(cls):
        super().__init_subclass__()
        # assert False, cls
        for body_part in cls.BODY_NAMES:
            property_name = f"{body_part}_mass"
            mass_property = property(
                fget=partial(cls.get_mass, body_part=body_part),
                fset=partial(cls._mass_setter, body_part),
            )
            setattr(cls, property_name, mass_property)

    def _update(self) -> None:
        """'Update' the model, if necessary, after a change has occured to the mass.

        TODO: Not sure if this is entirely correct
        """
        pass
        # self.model._compute_subtree()
        # self.model.forward()

    def reset_masses(self) -> None:
        """Resets the masses to their default values."""
        # NOTE: Use [:] to modify in-place, just in case there are any
        # pointer-shenanigans going on on the C side.
        self.model.body_mass[:] = self.default_masses
        # self.model._compute_subtree() #TODO: Not sure about this call
        # self.model.forward()

    def get_masses_dict(self) -> Dict[str, float]:
        return {
            body_name: self.model.body_masses[i]
            for i, body_name in enumerate(self.model.body_names)
        }

    def set_mass(self, **body_name_to_mass: Dict[str, Union[int, float]]) -> None:
        # Will raise an IndexError if the body part isnt found.
        # _set_mass(self, body_part=body_part, mass=mass)
        for body_part, mass in body_name_to_mass.items():
            idx = self.model.body_names.index(body_part)
            self.model.body_mass[idx] = mass

    def get_mass(self, body_part: str) -> float:
        # Will raise an IndexError if the body part isnt found.
        if body_part not in self.model.body_names:
            raise ValueError(
                f"No body named {body_part} in this mujoco model! (body names: "
                f"{self.model.body_names})."
            )
        idx = self.model.body_names.index(body_part)
        return self.model.body_mass[idx]

    def scale_masses(
        self,
        body_parts: List[str] = None,
        mass_scales: List[float] = None,
        **body_name_to_mass_scale,
    ) -> Dict[str, float]:
        """Scale the (original) mass of body parts of the Mujoco model.

        Returns a dictionary with the new masses.
        """
        new_masses: Dict[str, float] = {}
        body_parts = body_parts or []
        mass_scales = mass_scales or []
        body_name_to_mass_scale = body_name_to_mass_scale or {}

        self.reset_masses()

        body_name_to_mass_scale.update(zip(body_parts, mass_scales))

        for body_name, mass_scale in body_name_to_mass_scale.items():
            current_mass = self.get_mass(body_name)
            new_mass = mass_scale * current_mass
            self.set_mass(**{body_name: new_mass})

            new_masses[body_name] = new_mass

        # Not sure if we need to do this?
        self._update()
        return new_masses

    def get_and_modify_bodymass(self, body_name: str, scale: float):
        idx = self.model.body_names.index(body_name)
        temp = np.copy(self.model.body_mass)
        temp[idx] *= scale
        return temp

    @staticmethod
    def _mass_setter(body_part: str, env: MujocoEnv, mass: float) -> None:
        """Function used to set the mass of a body part. This is used as the setter of the
        generated `<body_part>_mass` properties.
        """
        # Will raise an IndexError if the body part isnt found.
        idx = env.model.body_names.index(body_part)
        env.model.body_mass[idx] = mass


# def _get_mass(env: MujocoEnv, /, body_part: str) -> float:
#     # Will raise an IndexError if the body part isnt found.
#     idx = env.model.body_names.index(body_part)
#     return env.model.body_mass[idx]
