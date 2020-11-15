from cwru import CWRU
@register_dataset("cwru")
class cwru_data(Dataset):
    num_classes = 16
    class_labels =['0.007-Ball',
                '0.007-InnerRace',
                '0.007-OuterRace12',
                '0.007-OuterRace3',
                '0.007-OuterRace6',
                '0.014-Ball',
                '0.014-InnerRace',
                '0.014-OuterRace6',
                '0.021-Ball',
                '0.021-InnerRace',
                '0.021-OuterRace12',
                '0.021-OuterRace3',
                '0.021-OuterRace6',
                '0.028-Ball',
                '0.028-InnerRace',
                'Normal'
    ]
    # window_size = 250
    # window_overlap = False

    def __init__(self, *args, **kwargs):
        super().__init__(cwru_data.num_classes, cwru_data.class_labels,
            *args, **kwargs)

    def process(self, data, labels):
        ...
        return super().process(data, labels)

    def load(self):
        self._cwru= CWRU("12DriveEndFault", '1797', 384)
        return self._cwru.X_train, self._cwru.y_train, self._cwru.X_test, self._cwru.y_test