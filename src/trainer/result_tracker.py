class Result_Tracker():
    def __init__(self, metric_name):
        self.metric_name = metric_name

    def init(self):
        if self.metric_name in ['rmse', 'mae']:
            init_value = 999
        elif ',' in self.metric_name:
            metric_name = self.metric_name.split(',')
            if metric_name == ['rmse', 'r2']:
                init_value = (999, -999)
            elif metric_name == ['spear', 'pear']:
                init_value = (-999, -999)
            elif metric_name == ['rocauc', 'aupr', 'acc']:
                init_value = (-999, -999, -999)
        else:
            init_value = -999
        return init_value

    def update(self, old_result, new_result):
        if self.metric_name in ['rmse', 'mae']:
            if new_result < old_result:
                return True
            else:
                return False
        elif ',' in self.metric_name:
            metric_name = self.metric_name.split(',')
            if metric_name == ['rmse', 'r2']:
                if new_result[0] <= old_result[0]:
                    return True
                else:
                    return False
            elif metric_name == ['spear', 'pear']:
                if new_result[0] >= old_result[0]:
                    return True
                else:
                    return False
            elif metric_name == ['rocauc', 'aupr', 'acc']:
                if new_result[0] >= old_result[0]:
                    return True
                else:
                    return False
        else:
            if new_result >= old_result:
                return True
            else:
                return False
