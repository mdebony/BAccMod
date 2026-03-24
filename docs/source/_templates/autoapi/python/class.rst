{% if obj.display %}
{% if is_own_page %}
{{ obj.short_name }}
{{ "=" * obj.short_name | length }}
{% endif %}

{% set visible_children = obj.children | selectattr("display") | rejectattr("short_name", "equalto", "__init__") | list %}
{% set own_page_children = visible_children | selectattr("type", "in", own_page_types) | list %}
{% set toctree_children = own_page_children | rejectattr("type", "equalto", "method") | list %}

{% if is_own_page and toctree_children %}
.. toctree::
   :hidden:

{% for child in toctree_children %}
   {{ child.include_path }}
{% endfor %}
{% endif %}

.. py:{{ obj.type }}:: {% if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}{% if obj.args %}({{ obj.args }}){% endif %}
{% for (args, return_annotation) in obj.overloads %}
   {{ obj.short_name }}{% if args %}({{ args }}){% endif %}
{% endfor %}
{{ '\n' }}

{% set visible_bases = [] %}
{% for base in obj.bases %}
{% if base != "object" %}
{% set _ = visible_bases.append(base) %}
{% endif %}
{% endfor %}
{% if visible_bases %}
   **Parent class{% if visible_bases|length > 1 %}es{% endif %}:** {% for base in visible_bases %}{% if base.startswith("baccmod.") %}:py:class:`~{{ base }}`{% else %}``{{ base }}``{% endif %}{% if not loop.last %}, {% endif %}{% endfor %}
{{ '\n' }}

{% endif %}
{% if obj.docstring %}
   {{ obj.docstring | indent(3) }}
{% endif %}
{% for obj_item in visible_children %}
{% if obj_item.type not in own_page_types and obj_item.type != "method" %}
   {{ obj_item.render() | indent(3) }}
{% endif %}
{% endfor %}

{% set own_page_or_inline_children = own_page_children | rejectattr("type", "equalto", "method") | list if is_own_page else [] %}
{% set visible_inline_methods = visible_children | selectattr("type", "equalto", "method") | list %}
{% set sorted_inline_methods = visible_inline_methods | sort(attribute="short_name") %}

{% if is_own_page and own_page_or_inline_children %}
{% set visible_attributes = own_page_or_inline_children | selectattr("type", "equalto", "attribute") | list %}
{% if visible_attributes %}
Attributes
----------
.. autoapisummary::

{% for attribute in visible_attributes %}
   {{ attribute.id }}
{% endfor %}
{% endif %}

{% set visible_exceptions = own_page_or_inline_children | selectattr("type", "equalto", "exception") | list %}
{% if visible_exceptions %}
Exceptions
----------
.. autoapisummary::

{% for exception in visible_exceptions %}
   {{ exception.id }}
{% endfor %}
{% endif %}

{% set visible_classes = own_page_or_inline_children | selectattr("type", "equalto", "class") | list %}
{% if visible_classes %}
Classes
-------
.. autoapisummary::

{% for klass in visible_classes %}
   {{ klass.id }}
{% endfor %}
{% endif %}

{% endif %}

{% if visible_inline_methods %}
Methods
-------
{% for method in sorted_inline_methods %}
{% if method.short_name[0] != "_" %}
* :py:meth:`~{{ method.id }}`
{% endif %}
{% endfor %}
{% for method in sorted_inline_methods %}
{% if method.short_name[0] == "_" %}
* :py:meth:`~{{ method.id }}`
{% endif %}
{% endfor %}

{% for method in sorted_inline_methods %}
{% if method.short_name[0] != "_" %}
   {{ method.render() | indent(3) }}
{% endif %}
{% endfor %}
{% for method in sorted_inline_methods %}
{% if method.short_name[0] == "_" %}
   {{ method.render() | indent(3) }}
{% endif %}
{% endfor %}
{% endif %}
{% endif %}
