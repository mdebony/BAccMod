{% set display_name = obj.name.split(".")[-1] %}

{{ display_name }}
{{ "=" * display_name|length }}

.. py:module:: {{ obj.name }}

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

{% if "function" in own_page_types %}
Functions
---------
.. toctree::
   :maxdepth: 1
   :glob:
   :titlesonly:

   {{ base }}/[a-z0-9_]*   {# package-root aliases, if any #}
{% endif %}

{% if obj.docstring %}
{{ obj.docstring }}
{% endif %}
