"""Task objects and gaze transitions for demonstration collection."""
from functools import wraps


# (manipulated/inspected actors, task goal/context actors). A list-valued task
# attribute is expanded into independent tracked entities after scene setup.
TASK_ROLES = {
    "adjust_bottle": (["bottle"], []),
    "beat_block_hammer": (["hammer"], ["block"]),
    "blocks_ranking_rgb": (["block1", "block2", "block3"], []),
    "blocks_ranking_size": (["block1", "block2", "block3"], []),
    "click_alarmclock": (["alarm"], []),
    "click_bell": (["bell"], []),
    "dump_bin_bigbin": (["deskbin"], ["dustbin"]),
    "grab_roller": (["roller"], []),
    "handover_block": (["box"], ["target_box"]),
    "handover_mic": (["microphone"], []),
    "hanging_mug": (["mug"], ["rack"]),
    "lift_pot": (["pot"], []),
    "move_can_pot": (["can"], ["pot"]),
    "move_pillbottle_pad": (["pillbottle"], ["pad"]),
    "move_playingcard_away": (["playingcards"], []),
    "move_stapler_pad": (["stapler"], ["pad"]),
    "open_laptop": (["laptop"], []),
    "open_microwave": (["microwave"], []),
    "pick_diverse_bottles": (["bottle1", "bottle2"], []),
    "pick_dual_bottles": (["bottle1", "bottle2"], []),
    "place_a2b_left": (["object"], ["target_object"]),
    "place_a2b_right": (["object"], ["target_object"]),
    "place_bread_basket": (["bread"], ["breadbasket"]),
    "place_bread_skillet": (["bread"], ["skillet"]),
    "place_burger_fries": (["hamburg", "frenchfries"], ["tray"]),
    "place_can_basket": (["can"], ["basket"]),
    "place_cans_plasticbox": (["object1", "object2"], ["plasticbox"]),
    "place_container_plate": (["container"], ["plate"]),
    "place_dual_shoes": (["left_shoe", "right_shoe"], ["shoe_box"]),
    "place_empty_cup": (["cup"], ["coaster"]),
    "place_fan": (["fan"], ["pad"]),
    "place_mouse_pad": (["mouse"], ["target"]),
    "place_object_basket": (["object"], ["basket"]),
    "place_object_scale": (["object"], ["scale"]),
    "place_object_stand": (["object"], ["displaystand"]),
    "place_phone_stand": (["phone"], ["stand"]),
    "place_shoe": (["shoe"], ["target_block"]),
    "press_stapler": (["stapler"], []),
    "put_bottles_dustbin": (["bottles"], ["dustbin"]),
    "put_object_cabinet": (["object"], ["cabinet"]),
    "rotate_qrcode": (["qrcode"], []),
    "scan_object": (["scanner", "object"], []),
    "shake_bottle": (["bottle"], []),
    "shake_bottle_horizontally": (["bottle"], []),
    "stack_blocks_three": (["block1", "block2", "block3"], []),
    "stack_blocks_two": (["block1", "block2"], []),
    "stack_bowls_three": (["bowl1", "bowl2", "bowl3"], []),
    "stack_bowls_two": (["bowl1", "bowl2"], []),
    "stamp_seal": (["seal"], ["target"]),
    "turn_switch": (["switch"], []),
}
MANUAL_TASKS = {"adjust_bottle", "handover_block", "stack_blocks_two"}


def resolve_actors(task, task_name):
    roles = []
    for attributes in TASK_ROLES[task_name]:
        role = {}
        for attribute in attributes:
            value = getattr(task, attribute)
            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError(f"empty task actor list: {task_name}.{attribute}")
                role.update({f"{attribute}[{i}]": actor for i, actor in enumerate(value)})
            else:
                role[attribute] = value
        roles.append(role)
    return roles


class ExecutionGazeRecipe:
    def __init__(self, task, task_name, subjects, goals):
        self.task, self.task_name = task, task_name
        self.subjects, self.goals = list(subjects), list(goals)
        self.names_by_identity = {id(actor): name for name, actor in {**subjects, **goals}.items()}
        self.active_by_arm = {}
        self.move_number = 0

    def install(self):
        # Attach semantic annotations to returned planned Action objects. This
        # cannot advance physics, change grasp selection, or move the camera.
        for method_name, operation in (("grasp_actor", "grasp"), ("place_actor", "place")):
            original = getattr(self.task, method_name)

            def wrap_method(original=original, operation=operation):
                @wraps(original)
                def annotated(*args, **kwargs):
                    actor = args[0] if args else kwargs["actor"]
                    result = original(*args, **kwargs)
                    name = self.names_by_identity.get(id(actor))
                    if result and name is not None:
                        for action in result[1]:
                            if action is not None:
                                action.av_semantic = (name, operation)
                    return result
                return annotated
            setattr(self.task, method_name, wrap_method())

        original_move = self.task.move

        @wraps(original_move)
        def move(actions_by_arm1, actions_by_arm2=None, save_freq=-1):
            if self.task.plan_success:
                self.before_move([actions_by_arm1, actions_by_arm2])
            return original_move(actions_by_arm1, actions_by_arm2, save_freq=save_freq)
        self.task.move = move

    def before_move(self, groups):
        av = self.task.active_view
        if av.mode == "external":
            return
        arms, semantics, action_kinds = [], [], []
        for group in groups:
            if not group or group[0] is None:
                continue
            side = str(group[0])
            arms.append(side)
            for action in group[1]:
                if action is None:
                    continue
                action_kinds.append(action.action)
                if hasattr(action, "av_semantic"):
                    name, operation = action.av_semantic
                    semantics.append((name, operation))
                    self.active_by_arm[side] = name
        self.move_number += 1
        operations = [op for _, op in semantics]
        if semantics:
            names = list(dict.fromkeys(name for name, _ in semantics))
            phase = "place" if "place" in operations else "grasp"
        else:
            names = list(dict.fromkeys(self.active_by_arm[side] for side in arms if side in self.active_by_arm))
            names = names or self.subjects
            phase = "gripper" if action_kinds and all(kind == "gripper" for kind in action_kinds) else "transport"
        if phase in {"place", "transport"}:
            names += [name for name in self.goals if name not in names]
        if phase == "place":
            # Preserve the object held by the other arm for handover/scanning
            # while one arm executes a placement or alignment action.
            names += [name for name in self.active_by_arm.values()
                      if name not in names and name in av.last_seen]
        # The other previously observed support is relevant when stacking.
        if phase == "place" and self.task_name.startswith("stack_"):
            names += [name for name in self.subjects if name not in names and name in av.last_seen]
        names = list(dict.fromkeys(names))
        av.focus(f"{phase}_{self.move_number:03d}", names, primary=names[0],
                 ee=list(dict.fromkeys(arms)), wait=False)

    def begin(self):
        self.task.active_view.focus("locate_task_objects", self.subjects, wait=True)


def configure_task(task, task_name):
    subjects, goals = resolve_actors(task, task_name)
    for name, actor in {**subjects, **goals}.items():
        task.active_view.register(name, actor)
    task.active_view.config["recipe"] = {
        "kind": "manual_phases" if task_name in MANUAL_TASKS else "execution_boundary",
        "subjects": list(subjects), "goals": list(goals),
    }
    if task_name in MANUAL_TASKS:
        return None
    recipe = ExecutionGazeRecipe(task, task_name, subjects, goals)
    recipe.install()
    return recipe
