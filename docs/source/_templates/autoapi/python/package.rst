:orphan:

{% set display_name = obj.name.split(".")[-1] %}

{{ display_name }}
{{ "=" * display_name|length }}

{% if obj.docstring %}
.. autoapi-nested-parse::
   {{ obj.docstring|indent(3) }}
{% endif %}
