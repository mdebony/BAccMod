{# Show only the last segment as the page title #}
{% set display_name = obj.name.split(".")[-1] %}

{{ display_name }}
{{ "=" * display_name|length }}

.. py:module:: {{ obj.name }}

{# Absolute base docname for this package, e.g. /autoapi/baccmod #}
{% set base = "/autoapi/" + obj.name.replace(".", "/") %}

Contents
--------
.. toctree::
   :maxdepth: 2
   :glob:
   :titlesonly:

   {{ base }}/*/index

{% if "class" in own_page_types %}
Classes
-------
.. toctree::
   :maxdepth: 1
   :glob:
   :titlesonly:

   {{ base }}/[A-Z]*
{% endif %}

{% if obj.docstring %}
{{ obj.docstring }}
{% endif %}
