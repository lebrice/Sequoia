
    def num_classes_in_task(self, task_id: int, train: bool) -> Union[int, List[int]]:
        """ Returns the number of classes in the given task. """
        increment = self.increment if train else self.test_increment
        if isinstance(increment, list):
            return increment[task_id]
        return increment

    def num_classes_in_current_task(self, train: bool = None) -> int:
        """ Returns the number of classes in the current task. """
        # TODO: Its ugly to have the 'method' tell us if we're currently in
        # train/eval/test, no? Maybe just make a method for each?
        return self.num_classes_in_task(self._current_task_id, train=train)

    def task_classes(self, task_id: int, train: bool) -> List[int]:
        """ Gives back the 'true' labels present in the given task. """
        start_index = sum(self.num_classes_in_task(i, train) for i in range(task_id))
        end_index = start_index + self.num_classes_in_task(task_id, train)
        if train:
            return self.class_order[start_index:end_index]
        else:
            return self.test_class_order[start_index:end_index]

    def current_task_classes(self, train: bool) -> List[int]:
        """ Gives back the labels present in the current task. """
        return self.task_classes(self._current_task_id, train)