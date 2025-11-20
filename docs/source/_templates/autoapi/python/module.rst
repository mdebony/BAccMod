{% set display_name = obj.name.split(".")[-1] %}

{{ display_name }}
{{ "=" * display_name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::
   {{ obj.docstring|indent(3) }}
{% endif %}

{# Children of this module that should be shown #}
{% set visible_children = obj.children | selectattr("display") | list %}
{% set visible_classes = visible_children | selectattr("type", "equalto", "class") | list %}
{% set visible_functions = visible_children | selectattr("type", "equalto", "function") | list %}
{% set visible_exceptions = visible_children | selectattr("type", "equalto", "exception") | list %}
{% set visible_attributes = visible_children | selectattr("type", "equalto", "data") | list %}

{% if visible_classes %}
Classes
-------
.. toctree::
   :maxdepth: 1
   :titlesonly:

{% for klass in visible_classes %}
   {{ klass.include_path }}
{% endfor %}
{% endif %}

{% if visible_functions %}
Functions
---------
.. toctree::
   :maxdepth: 1
   :titlesonly:

{% for func in visible_functions %}
   {{ func.include_path }}
{% endfor %}
{% endif %}

{% if visible_exceptions %}
Exceptions
----------
.. toctree::
   :maxdepth: 1
   :titlesonly:

{% for exc in visible_exceptions %}
   {{ exc.include_path }}
{% endfor %}
{% endif %}
