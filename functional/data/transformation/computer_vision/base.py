class AbstractTransformation(object):

    def apply(self, *args, **kwargs):
        """
        Apply the current transformation.

        Args:
            *args:
            **kwargs:
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)
