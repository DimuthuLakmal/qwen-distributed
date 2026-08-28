class InvalidRequestError(Exception):
    """The request body itself is malformed or missing required fields."""


class UnsupportedMediaTypeError(Exception):
    """An uploaded image is not a supported or decodable image type."""


class PayloadTooLargeError(Exception):
    """An uploaded image exceeds configured byte-size or pixel limits."""


class InvalidGenerationOptionsError(Exception):
    """Requested generation options are invalid or exceed server limits."""