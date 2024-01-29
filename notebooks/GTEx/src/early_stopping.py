import copy

class EarlyStopping():
  def __init__(self, patience=8, min_delta=-0.1, restore_best_weights=False, restore_app_weights=True):
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.restore_app_weights = restore_app_weights
    self.best_model = None
    self.app_model = None #the model that may perfrom a bit worse than the best on the test data but better on train data
    self.best_r2 = None
    self.app_r2 = None
    self.counter = 0
    self.status = ""

  def __call__(self, model, test_r2):
    if self.best_r2 == None:
      self.best_r2 = test_r2
      self.app_r2 = test_r2
      self.best_model = copy.deepcopy(model)
      self.app_model = copy.deepcopy(model)

    elif test_r2 - self.best_r2 >= 0:
      self.best_r2 = test_r2
      self.app_r2 = test_r2
      self.counter = 0
      self.best_model.load_state_dict(model.state_dict())
      self.app_model.load_state_dict(model.state_dict())

    elif test_r2 - self.best_r2 >= self.min_delta:
      self.counter = 0
      self.app_r2 = test_r2
      self.app_model.load_state_dict(model.state_dict())

    elif test_r2 - self.best_r2 < self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.status = f"Stopped on {self.counter}"
        if self.restore_app_weights:
          model.load_state_dict(self.app_model.state_dict())
        elif self.restore_best_weights:
          model.load_state_dict(self.best_model.state_dict())
        return True
    self.status = f"{self.counter}/{self.patience}"
    return False