import copy
import os
import time
import asyncio
from functools import partial
import threading
import ray

class AsySubmmitterConfig:
    def __init__(self):
        self.pending_length = 4
        #vicuna pending_length=4, baichuang pending_length=4, aquiq pending_length=5

    def get_pending_length(self):
        return self.pending_length

class RayTaskManager:
    def __init__(self):
        self.run = True
        self.pending_tasks_main =[]
        self.pending_tasks_aux = []
        self.wait_thread_main = threading.Thread(target=self.wait_for_all_tasks_main)
        self.wait_thread_main.start()
        self.wait_thread_aux = threading.Thread(target=self.wait_for_all_tasks_aux)
        self.wait_thread_aux.start()

    def apply(self, func, args=None, kwargs=None):
        kwargs = kwargs or {}
        args = args or ()
        task_id = func(*args, **kwargs)
        result = ray.get(task_id)
        return result
        
    def apply_async_main(self, func, args=None, kwargs=None, callback=None, callback_arg=None):
        kwargs = kwargs or {}
        args = args or ()
        task_id = func(*args, **kwargs)
        # print ("main pending tasks = {}".format(len(self.pending_tasks_main)))
        self.pending_tasks_main.append((task_id, callback, callback_arg))


    def apply_async_aux(self, func, args=None, kwargs=None, callback=None, callback_arg=None):
        kwargs = kwargs or {}
        args = args or ()
        task_id = func(*args, **kwargs)
        # print ("aux pending tasks = {}".format(len(self.pending_tasks_aux)))
        self.pending_tasks_aux.append((task_id, callback, callback_arg))
        
    def check_and_handle_tasks_main(self):
        if not self.pending_tasks_main:
            return
        task_ids = [task_id for task_id, _, _ in self.pending_tasks_main]
        ready_task_ids, _ = ray.wait(task_ids, num_returns=len(task_ids), timeout=0)
        for ready_task_id in ready_task_ids:
            for i, (task_id, callback, callback_arg) in enumerate(self.pending_tasks_main):
                if task_id == ready_task_id:
                    result = ray.get(task_id)
                    if callback:
                        callback(result, callback_arg)
                    self.pending_tasks_main.pop(i)
                    break

    def check_and_handle_tasks_aux(self):
        if not self.pending_tasks_aux:
            return
        task_ids = [task_id for task_id, _, _ in self.pending_tasks_aux]
        ready_task_ids, _ = ray.wait(task_ids, num_returns=len(task_ids), timeout=0)
        for ready_task_id in ready_task_ids:
            for i, (task_id, callback, callback_arg) in enumerate(self.pending_tasks_aux):
                if task_id == ready_task_id:
                    result = ray.get(task_id)
                    if callback:
                        callback(result, callback_arg)
                    self.pending_tasks_aux.pop(i)
                    break

    def wait_for_all_tasks_main(self):
        while self.run:
            self.check_and_handle_tasks_main()

    def wait_for_all_tasks_aux(self):
        while self.run:
            self.check_and_handle_tasks_aux()

    def get_main_pending_len(self):
        return len(self.pending_tasks_main)

    def get_aux_pending_len(self):
        return len(self.pending_tasks_aux)

    def stop_waiting(self):
        self.run = False