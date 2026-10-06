{% if obj.display %}
{% if is_own_page %}
.. orphan:
{% else %}
.. py:method:: {{ obj.id }}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}
{% for (args, return_annotation) in obj.overloads %}

            {{ obj.id }}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}
{% endfor %}
{% for property in obj.properties %}

   :{{ property }}:
{% endfor %}

{% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
{% endif %}
{% endif %}
{% endif %}
