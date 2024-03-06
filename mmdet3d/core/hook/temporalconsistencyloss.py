# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['ObjectTemporalConsistencyHook']


@HOOKS.register_module()
class ObjectTemporalConsistencyHook(Hook):
    """ """

    def __init__(self, loss_start_epoch=1, sclloss_start_epoch=1, depth_loss_end_epoch=999):
        super().__init__()
        self.loss_start_epoch = loss_start_epoch
        self.depth_loss_end_epoch = depth_loss_end_epoch
        self.sclloss_start_epoch = sclloss_start_epoch

    def set_loss_start_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.tcl_start_flag = flag
        else:
            runner.model.module.tcl_start_flag = flag

    def set_sclloss_start_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.scl_start_flag = flag
        else:
            runner.model.module.scl_start_flag = flag


    def set_depth_loss_end_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.depth_loss_type = flag
            print('depth_loss_end_epoch:{}, depth_loss_type: {}'.format(runner.epoch,runner.model.module.module.depth_loss_type))
        else:
            runner.model.module.depth_loss_type = flag
            print('depth_loss_end_epoch:{}, depth_loss_type: {}'.format(runner.epoch,runner.model.module.depth_loss_type))

    def before_run(self, runner):
        print('a before_run: runner.epoch:{} loss_start_epoch:{}'.format(runner.epoch,self.loss_start_epoch))
        self.set_loss_start_flag(runner, False)
        self.set_sclloss_start_flag(runner, False)
        print('b before_run: runner.epoch:{} loss_start_epoch:{}'.format(runner.epoch,self.loss_start_epoch))

    def before_train_epoch(self, runner):
        print('a before_train_epoch: runner.epoch:{} loss_start_epoch:{}'.format(runner.epoch,self.loss_start_epoch))
        if runner.epoch > self.loss_start_epoch:
            self.set_loss_start_flag(runner, True)
        print('b before_train_epoch: runner.epoch:{} loss_start_epoch:{}'.format(runner.epoch,self.loss_start_epoch))
        if runner.epoch > self.sclloss_start_epoch:
            self.set_sclloss_start_flag(runner, True)

        if runner.epoch > self.depth_loss_end_epoch:
            self.set_depth_loss_end_flag(runner, "none")

