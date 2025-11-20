{% set display_name = obj.name.split(".")[-1] %}

{{ display_name }}
{{ "=" * display_name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::
   {{ obj.docstring|indent(3) }}
{% endif %}

{# Subpackages and submodules of this package #}
{% set subpackages = obj.subpackages | selectattr("display") | list %}
{% set submodules = obj.submodules | selectattr("display") | list %}
{% set visible_submodules = (subpackages + submodules) | sort %}

{% if visible_submodules %}
Submodules
----------
.. toctree::
   :maxdepth: 2
   :titlesonly:

{% for sub in visible_submodules %}
   {{ sub.include_path }}
{% endfor %}
{% endif %}

{# Children defined directly in the package __init__ #}
{% set visible_children = obj.children | selectattr("display") | list %}
{% set visible_classes = visible_children | selectattr("type", "equalto", "class") | list %}
{% set visible_functions = visible_children | selectattr("type", "equalto", "function") | list %}

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
